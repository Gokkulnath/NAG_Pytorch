#!/usr/bin/env python
# coding: utf-8

# #### Get the Model weights and util Scripts

# In[ ]:


import sys
get_ipython().system('sudo apt  install protobuf-compiler')
get_ipython().system('{sys.executable} -m pip install protobuf')


# In[1]:


get_ipython().system('wget http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz')
get_ipython().system('tar -zxf *.tar.gz')
get_ipython().system('git clone https://github.com/vadimkantorov/caffemodel2pytorch.git')


# In[3]:


import torch


# In[10]:


get_ipython().system('{sys.executable} caffemodel2pytorch/caffemodel2pytorch.py vgg_face_caffe/VGG_FACE.caffemodel')

get_ipython().system('mv vgg_face_caffe/VGG_FACE.caffemodel.pt VGG_FACE.caffemodel.pt')
# Test Loaded Model

model = torch.load('VGG_FACE.caffemodel.pt')


# In[14]:


for k in model:
    print(model[k].shape)


# #### Clean up the Files

# In[16]:


get_ipython().system('rm -rf caffemodel2pytorch vgg_face_caffe*')

