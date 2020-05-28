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
import utils.statistics as stats
import utils.filterOperations as filters

# Get the soccer field mask
# Will be used to filter out unwanted segmented players by the teacher
# Can be replaced by any type of mask computation, or even a fixed mask
# The method is presented in the paper: 
# "A bottom-up approach based on semantics for the interpretation of the main camera stream in soccer games"
# by Anthony Cioppa, Adrien Deliège and Marc Van Droogenbroeck

def compute_soccer_field(frame):
	
	# Parameters for this function
	threshold_width = 10
	gaussian_blur_size = 9
	gaussian_blur_std = 1
	max_number_of_contour_edges = 20


	# Transform the image to HSV and apply gaussian blur
	frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
	frame_hsv_filtered = filters.gaussian_blur(frame_hsv, gaussian_blur_size, gaussian_blur_std)

	# Compute the histogram to recover the field peak (in green)
	histogram = stats.channel_histogram(frame_hsv_filtered, channel = 0)
	histogram_peak = stats.histogram_peak(histogram)

	# Compute the first rough mask
	thresholded_mask = filters.threshold(frame_hsv_filtered, channel = 0, threshmin = histogram_peak - threshold_width, threshmax =histogram_peak + threshold_width)

	# Apply some operations to filter the field mask
	# Delete the players, smooth the edges, etc.
	filtered_mask = filters.morphological_opening(thresholded_mask,15)
	filtered_mask = filters.morphological_closing(filtered_mask,17)

	contours = cv2.findContours(filtered_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]

	areas = []

	for contour in contours:
		areas.append(cv2.contourArea(contour))

	max_index = np.argmax(areas)
	
	# Approximation and convex Hull
	field_contour = contours[max_index]
	threshold_contour_approximation = 0
	
	filtered_mask.fill(0)
	cv2.drawContours(filtered_mask, [field_contour], 0, 255, -1)
	

	return filtered_mask

