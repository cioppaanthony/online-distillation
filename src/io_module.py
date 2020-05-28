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
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
import utils.multiprocess

def read(filename, queue_out_student, queue_out_teacher):

	print("Starting the video reading process")

	# Initializing the video
	video = cv2.VideoCapture(filename)

	# Check if the video is correctly opened
	if not video.isOpened():
		print("Error loading the video, make sure that the video exist and is readable")
		queue_out_student.put((None,False))
		queue_out_teacher.put((None,False))
		return

	# Reading the first frame for initializing the variables
	ret, frame = video.read()

	# Initializing the frame index counter
	frame_index = 0

	# Looping over all frames of the video
	while ret:

		# Send to teacher only if the queue is empty as we want to have 
		if queue_out_teacher.empty():
			# Send the original frame in the numpy format to the teacher
			queue_out_teacher.put((frame, True))

		# Transform the original frame to a torch format for the student
		tensor = utils.image.cvToTorch_(frame)

		# Send the torch tensor to the student
		queue_out_student.put((tensor, True))

		# Retrieve the next frame
		ret, frame = video.read()

		#print(queue_out_student.qsize(), " - ", queue_out_teacher.qsize())

	# Stopping signals
	queue_out_student.put((None, False))
	queue_out_teacher.put((None, False))

	print("Stopping the video reading process")
	time.sleep(15)

def write_student(save_path, queue_in, number_of_frames):

	print("Starting the student result writing process")

	# Initialize the variables for reading the queue
	results = None
	signal_continue = None

	# Get the first images from the queue
	results, signal_continue = queue_in.get()

	# Initialize the counter for saviing the results to different files
	frame_index = 0

	# Initialize the progress bar
	p_bar = tqdm(total=number_of_frames)

	# logging the time
	log_file = open(save_path + "log_time_student_segmentation.log", 'a')
	log_file.write("LOG FILE FOR THE STUDENT SEGMENTATION COMPUTATION TIME PER LOOP with a batch size of :" + str(results.size()[0]) + "\n")
	previous_time = time.time()

	# Save the results as long as they keep coming
	while signal_continue:
		
		# Transform the results to the numpy format
		frames = results.numpy()

		# Get the number of images to save
		number_to_save = frames.shape[0]

		# Loop over the batch
		for i in np.arange(number_to_save):

			# Write the images to the disk
			cv2.imwrite(save_path + "mask_" + str(frame_index) + ".png", frames[i])

			# Update the frame index
			frame_index += 1

		# Update the progress bar
		p_bar.update(number_to_save)

		# Retrieve the next results
		results, signal_continue = queue_in.get()

		# Logging the time
		log_file = open(save_path + "log_time_student_segmentation.log", 'a')
		log_file.write(str(time.time()-previous_time) + "\n")
		previous_time = time.time()
		log_file.close()

	# Close the progress bar
	p_bar.close()

	# Stopping signals
	print("Stopping the student result writing process")
	time.sleep(15)

def write_teacher(save_path, queue_in):

	print("Starting the teacher result writing process")

	# Initialize the variables for reading the queue
	original = None
	mask = None
	field = None
	signal_continue = None

	# Get the first images from the queue
	original, mask, field, signal_continue = queue_in.get()

	# Initialize the counter for saviing the results to different files
	frame_index = 0

	# Initialize the progress bar
	#p_bar = tqdm(total=10000)


	# logging the time
	log_file = open(save_path + "log_time_teacher_segmentation.log", 'a')
	log_file.write("LOG FILE FOR THE TEACHER SEGMENTATION COMPUTATION TIME PER LOOP" + "\n")
	previous_time = time.time()

	# Save the results as long as they keep coming
	while signal_continue:

		# Write the images to the disk
		cv2.imwrite(save_path + "original_" + str(frame_index) + ".png", original)
		cv2.imwrite(save_path + "mask_" + str(frame_index) + ".png", mask)
		cv2.imwrite(save_path + "field_" + str(frame_index) + ".png", field)

		# Update the frame index
		frame_index += 1

		# Update the progress bar
		#p_bar.update(1)

		# Retrieve the next results
		original, mask, field, signal_continue = queue_in.get()


		# Logging the time
		log_file = open(save_path + "log_time_teacher_segmentation.log", 'a')
		log_file.write(str(time.time()-previous_time) + "\n")
		previous_time = time.time()
		log_file.close()

	# Close the progress bar
	#p_bar.close()

	# Stopping signals
	print("Stopping the teacher result writing process")
	time.sleep(15)

def write_weights(save_path, queue_in):

	print("Starting the weights result writing process")

	# Initialize the variables for reading the queue
	new_parameters = None
	signal_continue = None

	# Get the first images from the queue
	new_parameters, signal_continue = queue_in.get()

	# Initialize the counter for saviing the results to different files
	frame_index = 0

	# Initialize the progress bar
	#p_bar = tqdm(total=10000)


	# logging the time
	log_file = open(save_path + "log_time_student_training.log", 'a')
	log_file.write("LOG FILE FOR THE STUDENT TRAINING COMPUTATION TIME PER LOOP" + "\n")
	previous_time = time.time()

	# Save the results as long as they keep coming
	while signal_continue:

		# Write the images to the disk
		torch.save(new_parameters, save_path + "weights_" + str(frame_index) + ".dat")

		# Update the frame index
		frame_index += 1

		# Update the progress bar
		#p_bar.update(1)

		# Retrieve the next results
		new_parameters, signal_continue = queue_in.get()

		# Logging the time
		log_file = open(save_path + "log_time_student_training.log", 'a')
		log_file.write(str(time.time()-previous_time) + "\n")
		previous_time = time.time()
		log_file.close()

	# Close the progress bar
	#p_bar.close()

	# Stopping signals
	print("Stopping the weights result writing process")
	time.sleep(15)