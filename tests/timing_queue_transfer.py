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
from multiprocessing import Process, Queue
from tqdm import tqdm
from queue import Empty

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="filepath to the input video")
parser.add_argument("-s", "--save", help="path_to_save")
parser.add_argument("-w", "--weights", help="filepath to the weights", default=None)

args = parser.parse_args()

def clear(q):
    try:
        while True:
            q.get_nowait()
    except Empty:
        pass

def read(Queue_out):

	print("Initializing function for reading frames")

	# Parsing of the arguments
	video_path = args.input

	device = torch.device("cuda:0")

	# Reading the video
	video = cv2.VideoCapture(video_path)

	if not video.isOpened():
		print("Error loading the video, make sure the video exists")
		sys.exit()

	# Reading video frames for the maximum number of matches

	while True:
		ret, frame = video.read()
		tensor = utils.image.cvToTorch(frame, device).to("cpu")[0]
		Queue_out.put(tensor)


def segment(Queue_in, Queue_out, Queue_weight):

	device = torch.device("cuda:1")

	print("Initializing function for segmenting frames")
	#Loading the network
	network = networks.tinynet.TinyNet(device).to(device)

	if args.weights is not None:
		network.load_state_dict(torch.load(args.weights))
	network.eval()

	tensor = None

	batch_size = 5
	counter = 0

	tensor = Queue_in.get()
	array_of_frames = torch.zeros(batch_size, tensor.size()[0], tensor.size()[1], tensor.size()[2], dtype = torch.uint8, device = "cpu")

	while True:
		if counter < batch_size:
			array_of_frames[counter] = tensor
			counter += 1
			tensor = Queue_in.get()
			continue
		if not Queue_weight.empty():
			new_parameters = Queue_weight.get()
			network.load_state_dict(new_parameters)
			network.eval()
		tensor_GPU = array_of_frames.to(device)
		tensor_GPU_norm = utils.image.normalize(tensor_GPU.type(torch.float))
		outputs = network.forward(tensor_GPU_norm)
		_,outputs = outputs.max(dim=1)
		outputs = outputs.type(torch.uint8)
		outputs_cpu = outputs.to("cpu")
		Queue_out.put(outputs_cpu)
		counter = 0

def write(Queue_in):

	print("Initializing function for writing frames")
	save_path = args.save
	tensor = None
	counter = 0

	pbar = tqdm(total=10000)

	while True:

		tensor = Queue_in.get()
		frame = tensor.numpy()
		for i in np.arange(frame.shape[0]):
			cv2.imwrite(save_path + str(counter) + ".png", frame[i])
			counter += 1
		pbar.update(5)
	pbar.close()

def weights_update(Queue_weight):

	print("Initializing function for updating weights")

	device = torch.device("cuda:2")

	#Loading the network
	network = networks.tinynet.TinyNet(device).to("cpu")
	
	if args.weights is not None:
		network.load_state_dict(torch.load(args.weights))
	network.eval()

	while True :
		time.sleep(4)
		weights = network.state_dict()
		clear(Queue_weight)
		Queue_weight.put(weights)

if __name__ == "__main__":

	print("Timing with multiple processes")

	queue_1 = Queue(maxsize = 10)
	queue_2 = Queue()
	queue_3 = Queue()

	read_process = Process(target=read, args=(queue_1,))
	segment_process = Process(target=segment, args=(queue_1,queue_2, queue_3))
	write_process = Process(target=write, args=(queue_2,))
	weight_process = Process(target=weights_update, args=(queue_3,))

	read_process.start()
	segment_process.start()
	write_process.start()
	weight_process.start()