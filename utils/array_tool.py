#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by XiaoYu on 18-3-1
"""
转换特殊类型的数据
"""
import numpy as np
import torch

def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch._TensorBase):
        return data.cpu().numpy()
    if isinstance(data, torch.autograd.Variable):
        return tonumpy(data.data)
# #测试tonumpy
# a = torch.zeros([2,2])
# print(a)
# print(tonumpy(a))
# b = torch.autograd.Variable(a)
# print(b)
# print(tonumpy(b))

def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch._TensorBase):
        tensor = data
    if isinstance(data, torch.autograd.Variable):
        tensor = data.data
    if cuda:
        tensor = tensor.cuda()
    return tensor

def tovariable(data):
    if isinstance(data, np.ndarray):
        return tovariable(totensor(data))
    if isinstance(data, torch._TensorBase):
        return torch.autograd.Variable(data)
    if isinstance(data, torch.autograd.Variable):
        return data
    else:
        raise ValueError("Unknow data type: %s, input should be {np.ndarray,Tensor,Variable}"%type(data))

def scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, torch._TensorBase):
        return data.view(1)[0]
    if isinstance(data, torch.autograd.Variable):
        return data.data.view(1)[0]

#测试scalar
a = np.array([
    5])
print(scalar(totensor(a)))
