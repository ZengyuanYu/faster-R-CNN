#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by XiaoYu on 18-3-1
import torch
from torch.utils.model_zoo import load_url
from torchvision import models

vgg16_caffe = load_url("https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth")
vgg16_caffe['classifier.0.weight'] = vgg16_caffe['classifier.1.weight']
vgg16_caffe['classifier.0.bias'] = vgg16_caffe['classifier.1.bias']
del vgg16_caffe['classifier.1.weight']
del vgg16_caffe['classifier.1.bias']

vgg16_caffe['classifier.3.weight'] = vgg16_caffe['classifier.4.weight']
vgg16_caffe['classifier.3.bias'] = vgg16_caffe['classifier.4.bias']
del vgg16_caffe['classifier.4.weight']
del vgg16_caffe['classifier.4.bias']

#模型存储
torch.save(vgg16_caffe, 'vgg16_caffe.pth')