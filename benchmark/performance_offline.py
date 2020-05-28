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

import sys
import torch
import cv2
import student
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import networks.maskrcnn
from networks.maskrcnn import COCODemo
from maskrcnn_benchmark.config import cfg
import utils.Graph as graph
import utils.field as field
import utils.image as process
import utils.filterOperations as filterOperations
import utils.folder_read_benchmark as folder_read


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

	dataset_images = list()
	dataset_labels = list()
	dataset_labels_border = list()

	device = torch.device(device_name)

	
	# --------------------------------------------------
	# Loading of the dataset from the pre-computed masks
	# --------------------------------------------------

	dataset_images, dataset_labels, dataset_labels_border = folder_read.load_from_folder(precomputed_images_path, True, device, dataset_images, dataset_labels, dataset_labels_border)

	# -----------------------------------
	# Training and testing of the network
	# -----------------------------------
	
	network = student.TinyNet(device, 2).to(device)

	# Loading of the initial weights
	# If none, evaluation of the scratch network (a bit useless)
	if weights_path is not None:
		network.load_state_dict(torch.load(weights_path, map_location=args.device))

	# Graph instantiation
	graph_path = save_path + "/graphs/"
	graph_metrics = graph.Graph(graph_path, "Offline")
	graph_metrics.set_title("Evolution on " + str(testingset_size) + " moving test images with border ("+ str(args.border[0]) + "," + str(args.border[1]) + ") of ")
	graph_metrics.set_names(["F1", "Jaccard", "Precision", "TPR", "FPR", "Accuracy"])

	counter_start_testset = trainingset_size
	counter_stop_testset = counter_start_testset + testingset_size

	network.eval()

	while True:

		images = dataset_images[counter_start_testset:counter_stop_testset]
		labels = dataset_labels_border[counter_start_testset:counter_stop_testset]

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


		graph_metrics.update(confusion_matrix.get_metrics(),True)


		
		counter_start_testset += s_sub
		counter_stop_testset = counter_start_testset + testingset_size

		torch.cuda.empty_cache()

		if counter_stop_testset >= len(dataset_images):
			print("End of the program")
			sys.exit(0)