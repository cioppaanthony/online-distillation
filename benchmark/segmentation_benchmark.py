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
import cv2
import glob
import torch
import student
import natsort
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import utils.image as process
import utils.field as field
import networks.maskrcnn
from networks.maskrcnn import COCODemo
from maskrcnn_benchmark.config import cfg
import utils.Graph as graph
import utils.filterOperations as filterOperations


def get_evaluation_number(path):

	parts = path.split("_", -1)
	return int(parts[-1].split(".",-1)[0])

def get_index_number(path):

	parts = path.split("_", -1)
	return int(parts[-1].split(".",-1)[0])



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


	device = torch.device(device_name)

	# -----------------------------
	# Loading the video and network
	# -----------------------------

	video = cv2.VideoCapture(video_path)
	if not video.isOpened():
		print("Error loading the video, make sure the video exists")
		sys.exit()

	network = student.TinyNet(device, 2).to(device)


	# Loading of the initial weights
	# If None, network was trained from scratch
	if weights_path is not None:
		network.load_state_dict(torch.load(weights_path, map_location=args.device))

	network.eval()

	weights_names = natsort.natsorted(glob.glob(save_path + "/weights/*.pt"))


	counter_read = 0.0
	counter_switch = 0
	index_switch = 0
	pbar_read = tqdm(total = video.get(cv2.CAP_PROP_FRAME_COUNT))

	ret, frame = video.read()
	frame_index = 0


	while ret:

		# Check when to switch the weights
		teacher_index = counter_read/t_sub
		if teacher_index >= trainingset_size + s_sub:

			if counter_switch >= t_sub * s_sub:
				if index_switch < len(weights_names):
					network.load_state_dict(torch.load(weights_names[index_switch], map_location=args.device))
					counter_switch = 0
					index_switch += 1
					print("Switched for : ", teacher_index)

		# Process all frames
		tensor = process.normalize(process.cvToTorch(frame, device))

		output = network.forward(tensor.type(torch.float)).squeeze_(0)
		_,output = output.max(dim=0)
		output = (1-output).type(torch.uint8)*255
		output = output.to("cpu").numpy()


		torch.cuda.empty_cache()

		# Save the output mask
		path_mask = save_path  + "/predictions_student/mask_" + str(int(counter_read)) + ".png"
		cv2.imwrite(path_mask, output)

		counter_read += 1
		counter_switch += 1
		ret, frame = video.read()
		frame_index += 1
		pbar_read.update(1)

	pbar_read.close()