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


class ResultMemory():
	
	def __init__(self):

		self.result_holder = []	# Holds a list of lists of values

	def add_result(self,new_result):

		if type(new_result) == None:
			print("WARGNING : Result to be saved are empty - No result saved at this step (In PyDS.Evaluation.Graphs.ResultMemory)")
			return 


		if type(new_result) == type(np.ndarray([0,0])):
			new_result = new_result.tolist()

		if type(new_result) == type(self.result_holder):
			if len(self.result_holder) == 0:
				self.result_holder.append(new_result)
			else : 
				if len(new_result) != len(self.result_holder[0]):
					print("Warning : Length of result is changed, results might be inconsistent or missing (In PyDS.Evaluation.Graphs.ResultMemory)")
					self.result_holder.append(new_result)
				else:
					self.result_holder.append(new_result)

		else :
			print("Type of new result is not a list, it is : ", type(new_result))


	def get_results(self):
		return self.result_holder

	def get_column(self, column_number):
		column_result = []

		for iterator in self.result_holder:
			column_result.append(iterator[column_number])

		return column_result

	def save_new(self, filename, new_result):
		log = open(filename, "a")

		for result in new_result:
			log.write(str(result))
			log.write("  ;  ")
		log.write("\n")
		log.close()