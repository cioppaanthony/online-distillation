"""
----------------------------------------------------------------------------------------
Copyright (c) 2020 - see AUTHORS file

This file is part of the ARTHuS software.

This program is free software: you can redistribute it and/or modify it under the terms 
of the GNU Affero General Public License as published by the Free Software Foundation, 
either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this 
program. If not, see < [ https://www.gnu.org/licenses/ | https://www.gnu.org/licenses/ ] >.
----------------------------------------------------------------------------------------
"""

import os
import sys
import torch
import cv2
import student
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import utils.field as field
import utils.Graph as graph
import utils.image as process
import utils.folder_read_benchmark as folder_read
import utils.filterOperations as filterOperations
import networks.maskrcnn
from networks.maskrcnn import COCODemo
from maskrcnn_benchmark.config import cfg


if __name__ == "__main__":

	# ------------------------
	# Parsing of the arguments
	# ------------------------

	from arguments_benchmark import args

	t_sub = args.teachersubsample
	s_sub = args.studentsubsample

	trainingset_size = args.datasetsize
	testingset_size = args.testingsetsize

	video_path = args.input
	save_path = args.save
	weights_path = args.weights

	precomputed_images_path = args.precomputed
	device_name = args.device

	learning_rate = args.learningrate

	print("Target classes: ", args.targetclass)

	# -------------------------
	# Create the saving folders
	# -------------------------

	saving_counter = 1
	while os.path.isdir(save_path + "/experiment_" + str(saving_counter)):
		saving_counter += 1
	save_path = save_path + "/experiment_" + str(saving_counter)

	os.mkdir(save_path)
	os.mkdir(save_path + "/graphs/")
	os.mkdir(save_path + "/predictions_student/")
	os.mkdir(save_path + "/predictions_teacher/")
	os.mkdir(save_path + "/videos/")
	os.mkdir(save_path + "/weights/")

	# -----------------------
	# Defining some variables
	# -----------------------

	dataset_images = list()
	dataset_labels = list()
	dataset_labels_border = list()

	device = torch.device(device_name)

	# If you need to compute the masks
	if precomputed_images_path is None:
		# Initializing the network
		config_file = "networks/weights/maskrcnn_weights.yaml"
		# update the config options with the config file
		cfg.merge_from_file(config_file)
		# manual override some options
		cfg.merge_from_list(["MODEL.DEVICE", device_name])
		coco_demo = COCODemo(
			cfg,
			min_image_size=800,
			confidence_threshold=0.7,
		)

		# --------------------------------------------
		# Reading the video and extracting the dataset
		# --------------------------------------------

		video = cv2.VideoCapture(video_path)
		if not video.isOpened():
			print("Error loading the video, make sure the video exists")
			sys.exit()

		counter_read = 0
		counter_save = 1

		pbar_read = tqdm(total = video.get(cv2.CAP_PROP_FRAME_COUNT))
		
		ret, frame = video.read()
		frame_index = 0

		while ret:

			# ----------------------------------------------
			# Only process and save one out of t_sub frames
			# This is the teacher segmentation by Mask R-CNN
			# Only on the soccer field
			# ----------------------------------------------
			if counter_read % t_sub == 0:

				# Compute the field mask
				field_mask = None
				if args.onfield is not None:
					field_mask = field.compute_soccer_field(frame)
				else:
					field_mask = np.zeros(frame[:,:,0].shape)+255

				# Get the predictions from Mask R-CNN
				predictions = coco_demo.run_on_opencv_image(frame)

				# Retrieve the object masks
				masks = predictions.get_field("mask").numpy()

				# Get the labels of these objects
				labels = predictions.get_field("labels")

				# And the bounding boxes
				boxes = predictions.bbox.numpy()

				# Print all segmented players whose bounding boxes intersect the field mask
				final = np.zeros(frame[:,:,0].shape)
				for mask, label, box in zip(masks, labels, boxes):
					if label in args.targetclass:
						covering_area = np.sum(mask[0,:,:]*field_mask)
						if covering_area == 0:
							continue
						final = np.logical_or(final,mask[0,:,:])
				final = final.astype("uint8")

				# Transform the image and the targets into the correct format for pytorch
				target = torch.from_numpy(1-final).type(torch.LongTensor).unsqueeze_(0).to("cpu")
				input_tensor = process.normalize(process.cvToTorch(frame, device)).to("cpu")
				dataset_images.append(input_tensor)
				dataset_labels.append(target)

				# Get a numpy image for saving the results to the save folder
				final = final*255

				save_path_original = save_path + "/predictions_teacher/original_" + str(counter_save) + ".png" 
				save_path_mask = save_path + "/predictions_teacher/mask_" + str(counter_save) + ".png" 
				save_path_field = save_path + "/predictions_teacher/field_" + str(counter_save) + ".png" 
				counter_save += 1
				cv2.imwrite(save_path_original, frame)
				cv2.imwrite(save_path_mask, final)
				cv2.imwrite(save_path_field, field_mask)

				# Get the evaluation borders on the players
				dataset_labels_border.append(folder_read.borderMask(final, int(args.border[0]), int(args.border[1])))




			pbar_read.update(1)
			counter_read += 1
			ret, frame = video.read()
			frame_index += 1

		pbar_read.close()

	else:
		dataset_images, dataset_labels, dataset_labels_border = folder_read.load_from_folder(precomputed_images_path, True, device, dataset_images, dataset_labels, dataset_labels_border)

	print("Teacher images stored")
	print("Number of images for the teacher : ", len(dataset_images))

	# -------------------------------------------------
	# Training and testing of the network
	# -------------------------------------------------

	# Building the network
	network = student.TinyNet(device, 2).to(device)

	# Loading of the initial weights
	# If none, trained from scratch
	if weights_path is not None:
		network.load_state_dict(torch.load(weights_path, map_location=args.device))

	# Definition of the optimizer
	optimizer = optim.Adam(network.parameters(), lr=learning_rate)

	# Definition of some indexes
	counter_start_trainset = 0
	counter_stop_trainset = counter_start_trainset + trainingset_size
	counter_start_testset = counter_stop_trainset
	counter_stop_testset = counter_start_testset + testingset_size

	# Graph instantiation
	graph_path = save_path + "/graphs/"
	graph_metrics = graph.Graph(graph_path, "Online") 
	graph_metrics.set_title("Evolution on " + str(testingset_size) + " moving test images with border ("+ str(args.border[0]) + "," + str(args.border[1]) + ") of ")
	graph_metrics.set_names(["F1", "Jaccard", "Precision", "TPR", "FPR", "Accuracy"])


	epoch_number = 0

	# Train the network and evaluate the different updates
	while True:

		# -------------
		# Training part
		# -------------
		print("Now training on: (", counter_start_trainset, ",", counter_stop_trainset, ")")

		images = dataset_images[counter_start_trainset:counter_stop_trainset]
		labels = dataset_labels[counter_start_trainset:counter_stop_trainset]

		# Get the weights for the class imbalance and update the criterion
		weights = folder_read.compute_weights(labels).to(device)
		criterion = torch.nn.CrossEntropyLoss(weights)

		network.train()

		pbar_train = tqdm(total = len(images))

		for tensor, label in zip(images, labels):

			tensor = tensor.to(device)
			label = label.to(device)
			outputs = network.forward(tensor)

			optimizer.zero_grad()
			loss = criterion(outputs, label)
			loss.backward()
			optimizer.step()

			tensor = tensor.to("cpu")
			label = label.to("cpu")

			pbar_train.update(1)
		pbar_train.close()

		# Save the network weights at each epoch
		network_weights = network.state_dict()
		path_save_weights = save_path + "/weights/weights_"+ str(epoch_number) + "_"  + str(counter_start_testset) + ".pt"
		torch.save(network_weights, path_save_weights)
		epoch_number += 1


		print("Trained on the current online dataset")

		# -------------
		# Testing part
		# -------------
		print("Now testing on: (", counter_start_testset, ",", counter_stop_testset, ")")

		images = dataset_images[counter_start_testset:counter_stop_testset]
		labels = dataset_labels_border[counter_start_testset:counter_stop_testset]

		network.eval()

		# Create the confusion matrix for this set
		confusion_matrix = folder_read.ConfusionMatrix(device)

		pbar_test = tqdm(total = len(images))

		for tensor, label in zip(images, labels):

			tensor = tensor.to(device)
			label = label.to(device)

			output = network.forward(tensor).squeeze_(0)
			_,output = output.max(dim=0)
			confusion_matrix.evaluate(1-output, label)

			tensor = tensor.to("cpu")
			label = label.to("cpu")
			output = output.to("cpu")
			pbar_test.update(1)
		pbar_test.close()
		print(confusion_matrix)
		graph_metrics.update(confusion_matrix.get_metrics(),True)

		# Update the indexes
		counter_start_trainset += s_sub
		counter_stop_trainset = counter_start_trainset + trainingset_size
		counter_start_testset = counter_stop_trainset
		counter_stop_testset = counter_start_testset + testingset_size

		torch.cuda.empty_cache()

		# End criterion
		if counter_stop_testset >= len(dataset_images):
			print("End of the program")
			sys.exit(0)