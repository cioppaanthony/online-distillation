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

import cv2
import time
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from networks.tinynet import TinyNet as TinyNet
from utils.statistics import compute_weights as compute_weights

def segment(network_name, weights_path, queue_from_stream, queue_to_student, queue_to_teacher, queue_to_save, device):


	# Creation of the network
	network = TinyNet(device, 2).to(device)

	# Loading of the initial weights
	if weights_path is not None:
		network.load_state_dict(torch.load(weights_path))

	network.eval()

	counter = 1

	# Loop for computing the segmantation maps on the frames
	while True:

		# Get the next frame from the queue
		input_tensor, frame_index = queue_from_stream.get()

		input_tensor = input_tensor.to(device)

		# Compute the forward pass

		output_tensor = network.forward(input_tensor)

		# Send the segmentation map to the saving path
		output_tensor = output_tensor.to("cpu")

		queue_to_save.put(output_tensor)

		# Check if some new weights are available in the queue
		if not queue_to_student.empty():
			print("Loading a new network")

			new_parameters = queue_to_student.get()
			network.load_state_dict(new_parameters)
			#network = network.to(device)
			network.eval()

		torch.cuda.empty_cache()

		queue_to_teacher.queue.clear()
		queue_to_teacher.put( (input_tensor.to("cpu"), frame_index) )
		
		print(network_name)
		counter += 1

def train(network_name, weights_path, queue_to_training, queue_to_student, queue_save_network ,training_set_size, device):



	# Creation of the pointer to the network
	network = None

	# Parsing of the network between the students
	network = TinyNet(device, 2).to(device)

	# Loading of the initial weights
	if weights_path is not None:
		network.load_state_dict(torch.load(weights_path))

	network.train()

	# Creation of the lists to encapsulate all the images and labels

	images = list()
	labels = list()


	learning_rate = 0.00001
	optimizer = optim.Adam(network.parameters(), lr=learning_rate)

	index = 0

	counter = 0


	while True:

		
		# First, retrieve all images inside of the queue
		number_to_retrieve = queue_to_training.qsize()

		if len(images) < 180:
			cv2.waitKey(1000)

		
		while number_to_retrieve > 0:

			counter += 1

			data = queue_to_training.get()

			image = data[0].to(device)
			label = data[1].to(device).unsqueeze_(0)

			# If the list is not full yet
			if len(images) < training_set_size:
				images.append(image)
				labels.append(label)
			# If the list is full, update the list
			else:
				images[index] = image
				labels[index] = label
				index = (index + 1)%len(images)

			number_to_retrieve -= 1

		if len(images) < 180:
			continue
		
		weights = compute_weights(labels).to(device)

		#	Defining the loss function
		criterion = torch.nn.CrossEntropyLoss(weights)

		# Then train on one epoch

		for tensor, label in zip(images, labels):

			# Computing the forward pass
			outputs = network.forward(tensor)

			# Zero the gradient parameters
			optimizer.zero_grad()

			# Computation of the loss
			loss = criterion(outputs, label)

			# Backward pass of the loss
			loss.backward()

			# Define the next step of the optimizer
			optimizer.step()

		network = network.to("cpu")
		new_parameters = network.state_dict()

		queue_to_student.queue.clear()
		queue_to_student.put(new_parameters)
		queue_save_network.put([new_parameters, counter])
		network = network.to(device)
		network.train()
		
		print("New set of weights **********************")