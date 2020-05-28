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

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import utils.ResultMemory as rm

class Graph:

	def __init__(self, folder, name):

		self.memory = rm.ResultMemory()
		self.folder = folder
		self.name = name
		self.filename_graph = self.folder + "/" + self.name + "_graph_"
		self.filename_metric = self.folder + "/" + self.name + "_metrics.log"
		self.results_name = None
		self.number_of_column = 0
		self.title = ""

	def set_title(self, title):
		self.title = title

	def set_names(self, new_names):
		self.results_name = new_names
		self.number_of_column = len(new_names)
		self.memory.save_new(self.filename_metric,self.results_name)

	def update(self, new_result, save = False):
		self.memory.add_result(new_result)
		if save:
			for i in np.arange(self.number_of_column):
				self.save_graph(i)
		self.save_metric(new_result)

	def save_graph(self,column_number): 

		data = self.memory.get_column(column_number)
		number_of_steps = np.arange(len(data))

		plt.figure(figsize=(12, 14))

		#Axis transforms
		ax = plt.subplot(111)    
		ax.spines["top"].set_visible(False)    
		ax.spines["bottom"].set_visible(False)    
		ax.spines["right"].set_visible(False)    
		ax.spines["left"].set_visible(False)    
		ax.get_xaxis().tick_bottom()    
		ax.get_yaxis().tick_left() 


		plt.suptitle(self.title + self.results_name[column_number], fontsize=17, ha="center")

		plt.plot(number_of_steps, data, lw=2.5) 


		plt.savefig((self.filename_graph +self.results_name[column_number] + ".png"), bbox_inches="tight") 

	def save_metric(self, new_results):
		self.memory.save_new(self.filename_metric, new_results)
