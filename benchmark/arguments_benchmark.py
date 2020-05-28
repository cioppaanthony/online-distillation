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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="filepath to the video to segment in real-time")
parser.add_argument("-s", "--save", help="path to the folder for saving the results")
parser.add_argument("-f", "--onfield", help="if activated, teacher will only segment players on the field", default=None)
parser.add_argument("-w", "--weights", help="filepath to the pre-trained weights of the student network, if left empty the student trains from scratch", default=None)
parser.add_argument("-p", "--precomputed", help="filepath to the precomputed images from the teacher", default=None)

parser.add_argument("-n", "--datasetsize", help="size of the dataset", type = int, default=67)
parser.add_argument("-t", "--testingsetsize", help="size of the test dataset", type = int, default=67)
parser.add_argument("-l", "--learningrate", help="learning rate", type = float, default=0.00001)
parser.add_argument("-b", "--border", help="The border to add to the groundtruth",nargs=2, default=(3,3))
parser.add_argument("-d", "--device", help="device", default="cuda:0")
parser.add_argument("-c", "--targetclass", help="class for which to do the segmentation, default=person", action='append', type=int)
parser.add_argument("-tsub", "--teachersubsample", help="subsampling for the teacher, number of frames to skip", type = int, default=75)
parser.add_argument("-ssub", "--studentsubsample", help="subsampling for the student training, number of teacher frames to skip", type = int, default=5)

args = parser.parse_args()
