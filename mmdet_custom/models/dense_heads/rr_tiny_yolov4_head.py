# -*- coding:utf-8 -*-
from collections import OrderedDict
import torch.nn as nn
import torch
from mmdet.models.dense_heads.yolo_head import YOLOV3Head
from mmdet_custom.models.utils import yolo_blocks as vn_layer
from mmdet.models.builder import HEADS

from mmcv.cnn import (ConvModule, bias_init_with_prob, constant_init, is_norm,
                      normal_init)

@HEADS.register_module()
class RRTinyYolov4Head(YOLOV3Head):
    def _init_layers(self):
        head = [
            OrderedDict([
                ('10_max', nn.MaxPool2d(2, 2)),
                ('11_conv', vn_layer.Conv2dBatchLeaky(self.in_channels[0], self.in_channels[0], 3, 1)),
                ('12_conv', vn_layer.Conv2dBatchLeaky(self.in_channels[0], self.out_channels[0], 1, 1)),
            ]),

            OrderedDict([
                ('13_conv', vn_layer.Conv2dBatchLeaky(self.in_channels[1], self.in_channels[0], 3, 1)),
                ('14_conv', nn.Conv2d(self.in_channels[0], self.num_anchors * self.num_attrib, 1)),
            ]),

            OrderedDict([
                ('15_convbatch', vn_layer.Conv2dBatchLeaky(self.in_channels[1], self.out_channels[1], 1, 1)),
                ('16_upsample', nn.Upsample(scale_factor=2)),
            ]),

            OrderedDict([
                ('17_convbatch', vn_layer.Conv2dBatchLeaky(self.out_channels[0]+self.out_channels[1], 256, 3, 1)),
                ('18_conv', nn.Conv2d(256, self.num_anchors * self.num_attrib, 1)),
            ]),
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in head])

    def init_weights(self):
        pass
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             normal_init(m, mean=0, std=0.01)
    #         if is_norm(m):
    #             constant_init(m, 1)

    def forward(self, feats):
        stem, extra_x = feats
        stage0 = self.layers[0](stem)
        head0 = self.layers[1](stage0)
        stage1 = self.layers[2](stage0)
        stage2 = torch.cat((stage1, extra_x), dim=1)
        head1 = self.layers[3](stage2)
        head = [head0, head1]  # 小特征图在前
        return tuple(head),

