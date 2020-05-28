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
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import argparse
import sys
import utils.image
import time
import cv2
import numpy as np
import networks.tinynet
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="filepath to the input video")
parser.add_argument("-mi", "--minimum", help="The minimum simultaneous transfer", type=int, default=1)
parser.add_argument("-ma", "--maximum", help="The maximum simultaneous transfer", type=int, default=25)
parser.add_argument("-d", "--device", help="device to test", default="cuda:0")
parser.add_argument("-nt", "--numbertests", help="number of tests for the mean time", type = int, default=200)
parser.add_argument("-w", "--weights", help="filepath to the weights", default=None)

args = parser.parse_args()

# Parsing of the arguments
video_path = args.input

device = torch.device(args.device)

# Reading the video
video = cv2.VideoCapture(video_path)

if not video.isOpened():
	print("Error loading the video, make sure the video exists")
	sys.exit()

# Reading video frames for the maximum number of matches

counter = 0
array_of_frames = None

while counter < args.maximum:

	ret, frame = video.read()

	
	tensor = utils.image.cvToTorch(frame, device).to("cpu")[0]

	if counter == 0:

		array_of_frames = torch.zeros(args.maximum, tensor.size()[0], tensor.size()[1], tensor.size()[2], dtype = torch.uint8, device = "cpu")
	array_of_frames[counter] = tensor
	counter += 1


#Loading the network
network = networks.tinynet.TinyNet(device).to(device)

if args.weights is not None:
	network.load_state_dict(torch.load(args.weights))
network.eval()
batch_sizes = np.arange(args.minimum, args.maximum + 1)

for batch_size in batch_sizes:

	mean_time_cpu_to_gpu = 0.0
	mean_time_segmentation = 0.0
	mean_time_training = 0.0
	mean_time_gpu_to_cpu = 0.0
	mean_time_gpu_to_cpu_2 = 0.0
	short_frames = array_of_frames[0:batch_size]

	learning_rate = 0.00001
	optimizer = optim.Adam(network.parameters(), lr=learning_rate)
	criterion = torch.nn.CrossEntropyLoss()

	start_total_time = time.time()
	for test in np.arange(args.numbertests):
		network.eval()
		start_time_cpu_to_gpu = time.time()
		short_frames_GPU = short_frames.to(device)
		stop_time_cpu_to_gpu = time.time()

		start_time_segmentation = time.time()
		short_frames_GPU_norm = utils.image.normalize(short_frames_GPU.type(torch.float))
		outputs = network.forward(short_frames_GPU_norm)
		#_,outputs = outputs.max(dim=1)
		#outputs = outputs.type(torch.uint8)
		stop_time_segmentation = time.time()

		mean_time_cpu_to_gpu += (stop_time_cpu_to_gpu - start_time_cpu_to_gpu)

		mean_time_segmentation += (stop_time_segmentation - start_time_segmentation)

		# Training time
		network.train()
		_,targets = outputs.max(dim=1)
		targets = (1-targets.type(torch.float)).type(torch.LongTensor).to(device)

		start_time_training = time.time()
		outputs_training = network.forward(short_frames_GPU_norm)
		optimizer.zero_grad()
		loss = criterion(outputs_training, targets)
		loss.backward()
		optimizer.step()
		stop_time_training = time.time()



		start_time_gpu_to_cpu = time.time()
		outputs_cpu = outputs.to("cpu")
		stop_time_gpu_to_cpu = time.time()
		"""
		start_time_gpu_to_cpu_2 = time.time()
		short_frames = short_frames.to("cpu")
		stop_time_gpu_to_cpu_2 = time.time()
		"""
		mean_time_training += (stop_time_training - start_time_training)
		mean_time_gpu_to_cpu += (stop_time_gpu_to_cpu - start_time_gpu_to_cpu)
		#mean_time_gpu_to_cpu_2 += (stop_time_gpu_to_cpu_2 - start_time_gpu_to_cpu_2)
	stop_total_time = time.time()


	mean_time_cpu_to_gpu = (mean_time_cpu_to_gpu/args.numbertests)*1000
	mean_time_training = (mean_time_training/args.numbertests)*1000
	mean_time_gpu_to_cpu = (mean_time_gpu_to_cpu/args.numbertests)*1000
	mean_time_gpu_to_cpu_2 = (mean_time_gpu_to_cpu_2/args.numbertests)*1000
	mean_time_segmentation = (mean_time_segmentation/args.numbertests)*1000
	mean_total_time = (stop_total_time - start_total_time)/args.numbertests *1000

	print("Mean time from cpu to gpu with batch size of (", batch_size, ") = ", mean_time_cpu_to_gpu, "ms")
	print("Mean time for segmentation with batch size of (", batch_size, ") = ", mean_time_segmentation, "ms")
	print("Mean time from gpu to cpu with batch size of (", batch_size, ") = ", mean_time_gpu_to_cpu, "ms")
	print("Mean time from gpu to cpu 2 with batch size of (", batch_size, ") = ", mean_time_gpu_to_cpu_2, "ms")
	print("Mean time Training with batch size of (", batch_size, ") = ", mean_time_training, "ms")
	print("Mean time from cpu to gpu with batch size of (", batch_size, ") per image = ", mean_time_cpu_to_gpu/batch_size, "ms")
	print("Mean time for segmentation with batch size of (", batch_size, ") per image = ", mean_time_segmentation/batch_size, "ms")
	print("Mean time from gpu to cpu with batch size of (", batch_size, ") per image = ", mean_time_gpu_to_cpu/batch_size, "ms")
	print("Mean time from gpu to cpu 2 with batch size of (", batch_size, ") per image = ", mean_time_gpu_to_cpu_2/batch_size, "ms")
	print("Mean time Training with batch size of (", batch_size, ") per image = ", mean_time_training/batch_size, "ms")
	print("Mean total time  with batch size of (", batch_size, ") = ", mean_total_time, "ms")
	print("Mean total time  with batch size of (", batch_size, ") per image = ", mean_total_time/batch_size, "ms")
	print("-------------------")
