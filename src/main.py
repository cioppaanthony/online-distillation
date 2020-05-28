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
import torch
import sys
import student
import teacher
import io_module
from multiprocessing import Process, Queue
from utils.statistics import video_number_of_frames


if __name__ == "__main__":




	# 	------------------------
	#	Parsing of the arguments
	# 	------------------------

	from arguments import args

	print("Target classes: ", args.targetclass)

	# -------------------------
	# Create the saving folders
	# -------------------------

	save_path = args.save
	saving_counter = 1
	while os.path.isdir(save_path + "/experiment_" + str(saving_counter)):
		saving_counter += 1
	save_path = save_path + "/experiment_" + str(saving_counter)

	os.mkdir(save_path)
	os.mkdir(save_path + "/predictions_student/")
	os.mkdir(save_path + "/predictions_teacher/")
	os.mkdir(save_path + "/videos/")
	os.mkdir(save_path + "/weights/")


	# 	--------------------------------------
	#	Define the GPU devices of each process
	# 	--------------------------------------

	# Select the device on which to perform the real-time segmentation of the student network
	device_student = torch.device(args.devicestudent)

	# Select the device on which to perform the groundtruth computation of the teacher network
	device_teacher = args.deviceteacher

	# Select the device on which to train online the student network
	device_training = torch.device(args.devicetraining)





	# 	-----------------------------------------------------------------
	#	Definition of the queues for the data transfers between processes
	# 	-----------------------------------------------------------------

	# Queue for storing the original frames in the numpy format for the teacher network
	queue_input_stream_numpy = Queue(maxsize=1)

	# Queue for storing the original frames in the torch format for the student network
	queue_input_stream_torch = Queue(maxsize=35)

	# Queue for storing the original frames and segmentation computed by the teacher for updating the online dataset
	queue_to_training = Queue(maxsize=100)

	# Queue for transfering the new trained weights to the real-time student
	queue_weights = Queue(maxsize=1)

	# Queue for saving the segmentation of the student network
	queue_save_student = Queue()

	# Queue for saving the segmentation of the teacher network
	queue_save_teacher = Queue()

	# Queue for saving the weights of the student network
	queue_save_weights = Queue()





	# 	---------------------------
	#	Definition of the processes
	# 	---------------------------

	# Process for reading the input video
	process_read = Process(target=io_module.read, args=(args.input, queue_input_stream_torch, queue_input_stream_numpy))

	# Process for the real-time student segmentation of the student network
	process_student = Process(target=student.segment, args=(queue_input_stream_torch, queue_save_student, queue_weights, device_student, args.batchsize, args.weights))

	# Process for the training of the real-time student segmentation network
	process_training = Process(target=student.train, args=(queue_to_training, queue_weights, queue_save_weights, args.datasetsize, device_training, args.weights))
	
	# Process for the teacher network
	process_teacher = Process(target=teacher.segment, args=(queue_input_stream_numpy, queue_to_training, queue_save_teacher, device_teacher, args.maskrcnnframerate, args.targetclass))

	# Process for saving the results of the fast segmentation network
	process_save_student = Process(target=io_module.write_student, args=(save_path + "/predictions_student/", queue_save_student, video_number_of_frames(args.input)))

	# Process for saving the results of the teacher network
	process_save_teacher = Process(target=io_module.write_teacher, args=(save_path + "/predictions_teacher/", queue_save_teacher))
	
	# Process for saving the results of the fast segmentation network
	process_save_weights = Process(target=io_module.write_weights, args=(save_path + "/weights/", queue_save_weights))



	# 	--------------------------------
	#	Starting the different processes
	# 	--------------------------------

	process_read.start()
	process_student.start()
	process_teacher.start()
	process_training.start()
	process_save_student.start()
	process_save_teacher.start()
	process_save_weights.start()


	process_save_student.join()
	process_save_teacher.join()
	process_save_weights.join()
	process_training.join()
	process_student.join()
	process_teacher.join()
	process_read.join()

	print("The program exited correctly")