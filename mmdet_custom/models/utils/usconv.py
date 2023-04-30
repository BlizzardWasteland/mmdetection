# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, CONV_LAYERS, NORM_LAYERS
from mmcv.runner import BaseModule
from mmcv.ops import DeformConv2d, deform_conv2d, ModulatedDeformConv2d, modulated_deform_conv2d, SyncBatchNorm, \
    DeformConv2dPack, ModulatedDeformConv2dPack
from torch.nn.modules.utils import _pair
import torch
import math
from mmcv.cnn import build_conv_layer, build_norm_layer
import numpy as np
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.models.utils import make_divisible

@CONV_LAYERS.register_module('USConv2d')
class USConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, depthwise=False, bias=True):
        super(USConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.depthwise = depthwise

    def forward(self, input): 
        self.groups = self.in_channels if self.depthwise else 1
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]

        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias

        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


def count_usconvNd_flops(m, x, y):
    x = x[0]
    kernel_ops = np.prod(list(m.kernel_size)) # Kw x Kh
    bias_ops = 1 if m.bias is not None else 0
    
    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    total_ops = y.nelement() * (m.in_channels // m.groups * kernel_ops + bias_ops)

    m.total_ops += torch.DoubleTensor([int(total_ops)])

def count_usconvNd_params(m, x, y):
    kernel_ops = np.prod(list(m.kernel_size)) # Kw x Kh
    m.total_params[0] = kernel_ops*m.in_channels*m.out_channels//m.groups



@NORM_LAYERS.register_module('USBN2d')
class USBatchNorm2d(_BatchNorm):
    def __init__(self,
                 *args,
                 bn_training_mode=True,
                 **kwargs):
        super(USBatchNorm2d, self).__init__(*args, **kwargs)
        self.bn_training_mode = bn_training_mode

    def forward(self, input):
        y = nn.functional.batch_norm(
            input,
            self.running_mean[:self.num_features],
            self.running_var[:self.num_features],
            self.weight[:self.num_features],
            self.bias[:self.num_features],
            self.training,
            self.momentum,
            self.eps)

        return y
    
    def train(self, mode=True):
        super(USBatchNorm2d, self).train(mode)
        if not mode and self.bn_training_mode:
            self.training = True


def count_usbn_flops(m, x, y):
    x = x[0]
    nelements = x.numel()
    # subtract, divide, gamma, beta
    total_ops = 2 * nelements
    m.total_ops += torch.DoubleTensor([int(total_ops)])

def count_usbn_params(m, x, y):
    m.total_params[0] = 2*y.shape[1]

def set_channel_ratio(layer, ratio, divisor):
    for module in layer.modules():
        if isinstance(module, nn.Conv2d):
            module.in_channels = make_divisible(module.weight.shape[1]*ratio, divisor)
            module.out_channels = make_divisible(module.weight.shape[0]*ratio, divisor)
        if isinstance(module, _BatchNorm):
            module.num_features = make_divisible(module.weight.shape[0]*ratio, divisor)
        if isinstance(module, nn.Linear):
            module.in_features = make_divisible(module.weight.shape[1]*ratio, divisor)
            module.out_features = make_divisible(module.weight.shape[0]*ratio, divisor)

def set_channels(layer, in_channels=None, out_channels=None):
    for module in layer.modules():
        if isinstance(module, nn.Conv2d):
            if in_channels:
                module.in_channels = in_channels
            if out_channels:
                module.out_channels = out_channels
        if isinstance(module, _BatchNorm):
            if in_channels:
                module.num_features = in_channels
            if out_channels:
                module.num_features = out_channels
        if isinstance(module, nn.Linear):
            if in_channels:
                module.in_features = in_channels
            if out_channels:
                module.out_features = out_channels
