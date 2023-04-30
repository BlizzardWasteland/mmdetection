# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule

from mmdet.models.builder import NECKS
from mmdet.models.utils import CSPLayer
from mmdet.models.necks import YOLOXPAFPN
from ..utils.usconv import set_channel_ratio, make_divisible, set_channels


@NECKS.register_module()
class SearchableYOLOXPAFPN(YOLOXPAFPN):
    def set_arch(self, arch, divisor=8):
        widen_factor_backbone = arch['widen_factor_backbone'][-len(self.in_channels):]
        in_channels = [make_divisible(c*alpha/0.5, divisor) for c,alpha in zip(self.in_channels, widen_factor_backbone)]
        out_channels = make_divisible(self.out_channels*arch['widen_factor_neck_out']/0.5, divisor)
        for idx in range(len(in_channels) - 1, 0, -1):
            _idx = len(in_channels)-1-idx
            set_channels(self.reduce_layers[_idx], 
                    in_channels[idx], in_channels[idx - 1])
            set_channel_ratio(self.top_down_blocks[_idx], 
                    widen_factor_backbone[idx - 1], divisor=divisor)
            set_channels(self.top_down_blocks[_idx].main_conv.conv, 
                    in_channels=in_channels[idx - 1] * 2)
            set_channels(self.top_down_blocks[_idx].short_conv.conv, 
                    in_channels=in_channels[idx - 1] * 2)

        for idx in range(len(in_channels) - 1):
            set_channels(self.downsamples[idx], 
                    in_channels[idx], in_channels[idx])
            set_channel_ratio(self.bottom_up_blocks[idx], 
                    widen_factor_backbone[idx+1], divisor=divisor)
            set_channels(self.bottom_up_blocks[idx].main_conv.conv, 
                    in_channels=in_channels[idx] * 2)
            set_channels(self.bottom_up_blocks[idx].short_conv.conv, 
                    in_channels=in_channels[idx] * 2)
        
        for i in range(len(in_channels)):
            set_channels(self.out_convs, in_channels[i], out_channels)
        print(self)
        print(in_channels)