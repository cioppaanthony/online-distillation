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
import torch
import numpy as np
import networks.maskrcnn
from networks.maskrcnn import COCODemo
from maskrcnn_benchmark.config import cfg
import utils.image
import utils.field
from arguments import args

def segment(queue_in, queue_out, queue_save, device, framerate=1, target_class=[1]):

	print("Starting the teacher segmentation process")

	# Define the weights to use
	config_file = "networks/weights/maskrcnn_weights.yaml"

	# Update the config options with the config file
	cfg.merge_from_file(config_file)

	# Manual override some options
	cfg.merge_from_list(["MODEL.DEVICE", device])

	# Create the maskrcnn model
	coco_demo = COCODemo(
		cfg,
		min_image_size=800,
		confidence_threshold=0.7,
	)

	# Initialise the variables that will be retrieved in the input queue
	signal_continue = None
	input_frame = None
	input_frame, signal_continue = queue_in.get()

	# Time counter for slowing the loop
	time_next_iteration = time.time() + framerate

	while signal_continue:

		# Compute the field mask
		field_mask = None
		if args.onfield is not None:
			field_mask = utils.field.compute_soccer_field(input_frame)
		else:
			field_mask = np.zeros(input_frame[:,:,0].shape)+255

		# Get the predictions from Mask-RCNN
		predictions = coco_demo.run_on_opencv_image(input_frame)

		# Retrieve the masks
		masks = predictions.get_field("mask").numpy()

		# Retrieve the labels
		labels = predictions.get_field("labels")

		# Create the final mask
		final = np.zeros(input_frame[:,:,0].shape)

		for mask, label in zip(masks, labels):

			# Only keep it if it is a person (if target_class=1)
			if not label in target_class:
				continue

			# Only keep if the it intersects the field
			covering_area = np.sum(mask[0,:,:]*field_mask)
			if covering_area == 0:
				continue

			# Since the mask filled all the requirements, we add it to the final result
			final = np.logical_or(final, mask[0,:,:])

		# Transform the image in the correct format
		final = final.astype("uint8")

		# Transform the data so they match the training specs
		target = torch.from_numpy(1-final).type(torch.uint8).unsqueeze_(0)
		input_tensor = utils.image.cvToTorch_(input_frame).unsqueeze_(0)

		# Send the data to the training process
		queue_out.put((input_tensor, target, True))

		# Rectify the data so it can be saved
		final = final * 255

		# Send the data to save to the writing process for the teacher
		queue_save.put((input_frame, final, field_mask, True))

		# Retrieve the next data
		input_frame, signal_continue = queue_in.get()

		# Slow down the loop to keep
		if time.time() > time_next_iteration:
			time_next_iteration += framerate
			continue
		time_to_sleep = time_next_iteration - time.time()
		time.sleep(time_to_sleep)
		time_next_iteration += framerate


	queue_out.put((None, None, False))
	queue_save.put((None, None, None, False))

	print("Stopping the teacher segmentation process")
	time.sleep(15)