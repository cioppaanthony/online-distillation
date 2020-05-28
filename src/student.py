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

import time
import torch
import torch.optim as optim
import utils.image
import utils.multiprocess
import utils.statistics
from networks.tinynet import TinyNet

def segment(queue_in, queue_out, queue_weights, device, batch_size=1, path_weights=None, framerate=0.040):

	print("Starting the real-time student segmentation process")

	# -----------------------------
	# Initialisation of the network
	# -----------------------------

	# Loading the real-time segmentation network
	network = TinyNet(device).to(device)

	# Load the pre-trained weights if provided
	if path_weights is not None:
		network.load_state_dict(torch.load(path_weights, map_location='cuda:1'))

	# Set the network for evaluation mode
	network.eval()

	# Initialise the variables that will be retrieved in the input queue
	signal_continue = None
	input_tensor = None
	input_tensor, signal_continue = queue_in.get()
	input_tensor_array = torch.zeros(batch_size, input_tensor.size()[0], input_tensor.size()[1], input_tensor.size(2), dtype=torch.uint8, device="cpu")


	# Time counter for slowing the loop
	framerate = framerate*batch_size
	time_next_iteration = time.time() + framerate

	while signal_continue:


		# Initialize the batch counter
		batch_counter = 0

		# Loop to construct the batch
		while batch_counter < batch_size and signal_continue:

			# Save the new image in the batch
			input_tensor_array[batch_counter] = input_tensor

			# Update the counter
			batch_counter += 1

			# Get the next image
			input_tensor, signal_continue = queue_in.get()

		# Once the batch is complete, check if new weights are available
		if not queue_weights.empty():
			new_parameters = queue_weights.get()
			network.load_state_dict(new_parameters)
			network.eval()

		# Send the data to the device as a byte tensor to optimize transfer speed
		input_tensor_GPU = input_tensor_array.to(device)

		# Normalize the data on the GPU before the forward pass
		input_tensor_GPU = utils.image.normalize_(input_tensor_GPU.type(torch.float))

		# Compute the forward pass of the network
		output_tensor = network.forward(input_tensor_GPU)

		# Binarize the output
		_, output_tensor = output_tensor.max(dim=1)

		# Transfor it in the right format
		output_tensor = (1-output_tensor)*255

		# Transform to byte tensor for transfer speed optimization
		output_cpu = output_tensor.type(torch.uint8).to("cpu")

		# Send the segmentation maps to the process responsible for saving them to the disk
		queue_out.put((output_cpu, True))

		# Slow down the loop to keep
		if time.time() > time_next_iteration:
			time_next_iteration += framerate
			continue
		time_to_sleep = time_next_iteration - time.time()
		if time_to_sleep > 0:
			time.sleep(time_to_sleep)
		time_next_iteration += framerate

	# Send a stop signal to the process writing the results
	queue_out.put((None, False))

	print("Stopping the real-time student segmentation process")
	time.sleep(15)

def train(queue_in, queue_out, queue_save, datasetsize, device, path_weights=None):

	print("Starting the real-time student training process")

	# -----------------------------
	# Initialisation of the network
	# -----------------------------

	# Loading the real-time segmentation network
	network = TinyNet(device).to(device)

	# Load the pre-trained weights if provided
	if path_weights is not None:
		network.load_state_dict(torch.load(path_weights, map_location='cuda:1'))

	# Set the network for training mode
	network.train()

	# Initialize the data structure to hold the online dataset
	images = list()
	labels = list()

	# Initialize the learning parameters
	learning_rate = 0.0001
	optimizer = optim.Adam(network.parameters(), lr=learning_rate)

	# Initialise the variables that will be retrieved in the input queue
	signal_continue = None
	input_tensor = None
	label = None
	input_tensor, label, signal_continue = queue_in.get()

	# Initialize the index for storing the next frame and the frame_counter
	frame_counter = 0
	next_index = 0

	while signal_continue:

		# Retrieve all images inside of the queue
		# First, we get the number of images inside of the queue
		number_to_retrieve = queue_in.qsize()

		# Retrieve the data
		while number_to_retrieve > 0 and signal_continue:

			# Update the frame counter
			frame_counter += 1

			# Get the next frame and label to add to the dataset
			label_device = label.to(device).type(torch.LongTensor)
			input_tensor = utils.image.normalize_(input_tensor.to(device).type(torch.float))

			# If the list is not full yet
			if len(images) < datasetsize:
				# Append the data to the list
				images.append(input_tensor)
				labels.append(label_device)

			# If the list is full, update the list
			else:
				images[next_index] = input_tensor
				labels[next_index] = label_device
				next_index = (next_index+1)%datasetsize

			number_to_retrieve -= 1

			# Read the next data
			input_tensor, label, signal_continue = queue_in.get()

		# Wait for the dataset to be complete to start the training to avoid overfitting
		if len(images) < datasetsize:
			continue

		# Compute the new set of weights
		weights = utils.statistics.compute_weights(labels).to(device)

		# Define the loss function
		criterion = torch.nn.CrossEntropyLoss(weights)

		# Train on one epoch
		for tensor, target in zip(images, labels):
			# Computing the forward pass
			outputs = network.forward(tensor)

			# Zero the gradient parameters
			optimizer.zero_grad()

			# Computation of the loss
			loss = criterion(outputs, target.to(device))

			# Backward pass of the loss
			loss.backward()

			# Define the next step of the optimizer
			optimizer.step()

		# Send the new parameters to the student network
		# Send the network to the cpu to retrieve the weights
		network = network.to("cpu")

		# Get the parameters of the trained network
		new_parameters = network.state_dict()

		# Clear the queue
		if not queue_out.empty():
			queue_out.get()

		# Put the new parameters in the queue for the student
		queue_out.put(new_parameters)

		# Put the new parameters in the queue for saving to the disk
		queue_save.put((new_parameters, True))

		# Put the network back on the GPU
		network = network.to(device)

	# Stopping signals
	queue_save.put((None,False))

	print("Stopping the real-time student training process")
	time.sleep(15)