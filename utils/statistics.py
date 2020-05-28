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
import numpy as np
import torch

# Get the histogram for one channel (in this case, the hue)
def channel_histogram(img, channel = 0):
	histogram = np.histogram(img[:,:,channel],np.arange(257))
	histogram_list = [histogram[0], np.arange(256)]
	return histogram_list

# Get the peak in the histogram
def histogram_peak(histogram):
	return histogram[1][np.argmax(histogram[0])]


# Compute the weights for the class imbalance between the class and the background
def compute_weights(labels):

	total = 0
	total_foreground = 0

	for label in labels:

		total_foreground += torch.sum(label == 0).to("cpu").numpy()
		total += label.size()[1] * label.size()[2]

	percentage = total_foreground/total

	weights = torch.Tensor(2)
	weights[0] = 1/percentage
	weights[1] = 1/(1-percentage)

	return weights

# Load a video and get its number of frames
def video_number_of_frames(video_path):

	video = cv2.VideoCapture(video_path)

	# Check if the video is correctly opened
	if not video.isOpened():
		print("Error loading the video, make sure that the video exist and is readable")
		return 0

	return int(video.get(cv2.CAP_PROP_FRAME_COUNT))