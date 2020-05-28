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


def morphological_opening(img, morph_size):
	kernel = np.ones((morph_size,morph_size),np.uint8) 
	return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def morphological_closing(img, morph_size):
	kernel = np.ones((morph_size,morph_size),np.uint8) 
	return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def morphological_dilation(img, morph_size):
	kernel = np.ones((morph_size,morph_size),np.uint8) 
	return cv2.dilate(img, kernel)

def morphological_erosion(img, morph_size):
	kernel = np.ones((morph_size,morph_size),np.uint8) 
	return cv2.erode(img, kernel)

def gaussian_blur(img, kernel_size, std):
	return cv2.GaussianBlur(img,(kernel_size,kernel_size),std)

def threshold(img, channel = 0, threshmin=None, threshmax = None):
	mask_1 = np.where(img[:,:,channel] > threshmin, 1, 0)
	mask_2 = np.where(img[:,:,channel] < threshmax, 1, 0)
	mask = (mask_1 == 1) & (mask_2 == 1)
	mask = mask.astype("uint8")
	mask[mask == 1] = 255
	return mask