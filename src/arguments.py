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
parser.add_argument("-i", "--input", help="filepath to the video to segment in real-time", required=True)
parser.add_argument("-s", "--save", help="path to the folder for saving the segmentation results", required=True)
parser.add_argument("-f", "--onfield", help="if activated, teacher will only segment players on the field", default=None)
parser.add_argument("-w", "--weights", help="filepath to the pre-trained weights of the student network, if left empty the student trains from scratch", default=None)


parser.add_argument("-dt", "--deviceteacher", help="device on which to run the teacher network", default="cuda")
parser.add_argument("-ds", "--devicestudent", help="device on which to run the student network", default="cuda:1")
parser.add_argument("-do", "--devicetraining", help="device on which to run the training of the student network", default="cuda:2")


parser.add_argument("-c", "--targetclass", help="class for which to do the segmentation, default=person", action='append', default=[1])
parser.add_argument("-b", "--batchsize", help="batchsize for the student network", type = int, default=5)
parser.add_argument("-d", "--datasetsize", help="size of the online dataset", type = int, default=67)
parser.add_argument("-m", "--maskrcnnframerate", help="time for the teacher to subsample images (in seconds)", type = int, default=3)
args = parser.parse_args()
