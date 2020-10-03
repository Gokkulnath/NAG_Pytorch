#!/usr/bin/env python
# coding: utf-8
'''
Script to Convert the Caffenet Model to Pytorch: For VGG-F Model Weight Conversion
'''


# #### Get the Model weights and util Scripts
import sys
from IPython import get_ipython

get_ipython().system('sudo apt  install protobuf-compiler')
get_ipython().system('{sys.executable} -m pip install protobuf')
get_ipython().system('wget http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz')
get_ipython().system('tar -zxf *.tar.gz')
get_ipython().system('git clone https://github.com/vadimkantorov/caffemodel2pytorch.git')

import torch

get_ipython().system('{sys.executable} caffemodel2pytorch/caffemodel2pytorch.py vgg_face_caffe/VGG_FACE.caffemodel')
get_ipython().system('mv vgg_face_caffe/VGG_FACE.caffemodel.pt VGG_FACE.caffemodel.pt')
# Test Loaded Model
model = torch.load('VGG_FACE.caffemodel.pt')

for k in model:
    print(model[k].shape)
get_ipython().system('rm -rf caffemodel2pytorch vgg_face_caffe*')

