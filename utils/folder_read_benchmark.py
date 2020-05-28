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
import cv2
import glob
import torch
import natsort
import numpy as np
from tqdm import tqdm
import utils.filterOperations as filterOperations

from arguments_benchmark import args

def load_from_folder(directory, normalized=True, device="cpu", append_original=None, append_mask=None, append_border=None):

	# Get all file names
	filename_list_original = natsort.natsorted(glob.glob(directory + "/original_*.png"))
	filename_list_masks = natsort.natsorted(glob.glob(directory + "/mask_*.png"))

	if append_mask == None or append_original == None:
		append_original = list()
		append_mask = list()

	pbar = tqdm(total = len(filename_list_original))

	for file_original, file_mask in zip(filename_list_original, filename_list_masks):

		tmp_image = cv2.imread(file_original, cv2.IMREAD_COLOR)
		tmp_label = cv2.imread(file_mask, cv2.IMREAD_GRAYSCALE)

		if append_border is not None:
			append_border.append(borderMask(tmp_label, int(args.border[0]), int(args.border[1])))

		torch_image = torch.from_numpy(tmp_image).to(device).type(torch.float).transpose(0,2).transpose(1,2)
		torch_label = torch.from_numpy(tmp_label).to(device).type(torch.float)
		torch_label = 1-(torch_label/255)
		torch_label = torch_label.type(torch.long).unsqueeze_(0)

		if normalized:
			torch_image = ((torch_image*2)/255) -1

		append_original.append(torch_image.unsqueeze_(0).to("cpu"))
		append_mask.append(torch_label.to("cpu"))

		pbar.update(1)
	pbar.close()

	return append_original, append_mask, append_border

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

class ConfusionMatrix:

	def __init__(self, device):

		self.TP = 0
		self.FP = 0
		self.FN = 0
		self.TN = 0

		self.ones = None 
		self.zeros = None
		self.device = device

	def evaluate(self, mask, groundtruth):

		if self.ones is None or self.zeros is None:
			self.update(groundtruth.size()[0], groundtruth.size()[1])

		self.ones = self.ones.to(self.device)
		self.zeros = self.zeros.to(self.device)

		TP_mask = torch.where((mask == 1) & (groundtruth == 1), self.ones, self.zeros)
		FP_mask = torch.where((mask == 1) & (groundtruth == 0), self.ones, self.zeros)
		FN_mask = torch.where((mask == 0) & (groundtruth == 1), self.ones, self.zeros)
		TN_mask = torch.where((mask == 0) & (groundtruth == 0), self.ones, self.zeros)
		
		self.TP += torch.sum(TP_mask)
		self.FP += torch.sum(FP_mask)
		self.FN += torch.sum(FN_mask)
		self.TN += torch.sum(TN_mask)

		self.ones = self.ones.to("cpu")
		self.zeros = self.zeros.to("cpu")

	def update(self, height, width):

		self.ones = torch.ones((height, width), dtype=torch.float32)
		self.zeros = torch.zeros((height, width), dtype=torch.float32)
	
	def F1(self):

		return ((2*self.TP)/(2*self.TP + self.FP + self.FN)).to("cpu").numpy()

	def Jaccard(self):

		return ((self.TP)/(self.TP + self.FP + self.FN)).to("cpu").numpy()

	def Accuracy(self):
		return ((self.TP + self.TN)/(self.TP + self.FP + self.FN + self.TN)).to("cpu").numpy()

	def Precision(self):

		return ((self.TP)/(self.TP + self.FP)).to("cpu").numpy()

	def TPR(self):

		return ((self.TP)/(self.TP + self.FN)).to("cpu").numpy()

	def FPR(self):

		return (1-((self.TN)/(self.TN+self.FP))).to("cpu").numpy()

	def get_metrics(self):
		return [self.F1(), self.Jaccard(), self.Precision(), self.TPR(), self.FPR(), self.Accuracy()]

	def __str__(self):

		print("Jaccard : ", self.Jaccard())
		print("F1 : ", self.F1())
		print("Precision :", self.Precision())
		print("TPR : ", self.TPR())
		print("FPR : ", self.FPR())
		print("Accuracy : ", self.Accuracy())
		print(self.TP, " - ", self.FP)
		print(self.FN, " - ", self.TN)

		return "-----"

def borderMask(mask, inner_border_size, outer_border_size):

	mask_eroded = filterOperations.morphological_erosion(mask, 2*inner_border_size+1)
	mask_dilated = filterOperations.morphological_dilation(mask, 2*outer_border_size+1)
	border = mask_dilated - mask_eroded
	out = np.abs(mask-0.5*border)
	out_tensor = torch.from_numpy(out).type(torch.float)/255
	return out_tensor