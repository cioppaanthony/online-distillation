#!/bin/bash
set -euf -o pipefail

echo --------------------------------
echo APT-GET
echo --------------------------------
apt-get -y update
apt-get -y upgrade
apt-get -y install htop

echo --------------------------------
echo APT-GET PYTHON
echo --------------------------------
apt-get -y install python-pip
apt-get -y install python3-pip python3-dev
apt-get -y install python3-tk
apt-get -y install libglib2.0-0
apt-get -y install libsm6 libxext6 libxrender-dev

echo --------------------------------
echo PIP INSTALL FOR PYTHON3
echo --------------------------------
python3 -m pip install --upgrade pip
python3 -m pip install numpy==1.16.3
python3 -m pip install torch==1.0.1.post2
python3 -m pip install torchvision==0.2.2
python3 -m pip install tqdm==4.23.1
python3 -m pip install matplotlib==3.0.3
python3 -m pip install opencv-python-headless==4.1.2.30
python3 -m pip install opencv-contrib-python-headless==4.1.2.30
python3 -m pip install natsort==6.0.0
python3 -m pip install imgaug==0.2.8
python3 -m pip install cython==0.29.7
python3 -m pip install ninja==1.9.0
python3 -m pip install yacs
python3 -m pip install ipython
python3 -m pip install scikit-image

echo --------------------------------
echo PIP INSTALL FOR PYTHON2
echo required -> external libraries
echo --------------------------------
python2 -m pip install --upgrade pip
python2 -m pip install numpy==1.16.3
python2 -m pip install matplotlib==2.2.4
python2 -m pip install cython==0.29.7
python2 -m pip install ninja==1.9.0
python2 -m pip install yacs 
python2 -m pip install ipython



echo --------------------------------
echo EXTERNAL LIBRARIES
echo --------------------------------
cd external_libraries

echo --------------------------------
echo COCO API
echo --------------------------------
cd cocoapi/PythonAPI/
make
python3 setup.py build_ext install

echo --------------------------------
echo Torchvision
echo --------------------------------
cd ../../
cd vision/
python3 setup.py install

echo --------------------------------
echo MaskRCNN Benchmark
echo --------------------------------
cd ../
cd maskrcnn-benchmark
python3 setup.py build develop

echo --------------------------------
echo Setting up maskrcnn
echo --------------------------------
cd ../
python3 setup_maskrcnn.py
