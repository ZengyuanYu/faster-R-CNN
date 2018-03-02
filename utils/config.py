#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by XiaoYu on 18-3-2
from pprint import pprint

"""
在训练的过程中提供配置，包括图片尺寸，参数和训练模型
使用方式：命令行后面加上
eg. --test_num=1000, --voc-data-dir='./data/'
"""
class Config:
    #数据读取位置
    voc_data_dir = '/home/administrator/yu/simple-faster-rcnn-pytorch/data/VOCdevkit/VOC2007/'

    #image size
    min_size = 600
    max_size = 1000
    num_workers = 8
    test_num_workers = 8

    #L1_smooth_loss的sigma
    rpn_sigma = 3.
    roi_sigma = 1.

    #优化器参数
    weight_decay = 0.0005
    lr_decay = 0.1
    lr = 1e-3

    #visdom可视化
    env = 'faster-rcnn' #visdom env
    port = 8097
    plot_every = 40

    #预置参数
    data = 'voc'
    pretrained_model = 'vgg16'
    epoch = 14

    use_adam = False
    use_chainer = False
    use_drop = False

    debug_file = '/tmp/debugf'
    test_num = 10000
    load_path = None

    caffe_pretrain = False #默认不用caffe模型而用torchvision
    caffe_pretrain_path = 'checkpoints/vgg16-caffe.pth'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('Unknow Options: "--%s"'%k)
            setattr(self, k, v)
        
        print('=========use config=========')
        pprint(self._state_dict())
        print('============end=============')
        
    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

opt = Config()

# print(opt.epoch)