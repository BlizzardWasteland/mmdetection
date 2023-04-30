# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm, BatchNorm2d

from mmdet.models.builder import BACKBONES
from mmdet.models.utils import CSPLayer
from mmdet.models.backbones.csp_darknet import Focus, SPPBottleneck

@BACKBONES.register_module()
class SearchableCSPDarknet(BaseModule):
    arch_settings = {
        'P5': [[64, 128, 9, True, False], [128, 256, 9, True, False],
               [256, 512, 9, True, False], [512, 1024, 9, False, True]],
    }
    def __init__(self,
                 arch='P5',
                 deepen_factor=[1.0]*4,
                 widen_factor=[1.0]*5,
                 out_indices=(2, 3, 4),
                 frozen_stages=-1,
                 use_depthwise=False,
                 arch_ovewrite=None,
                 spp_kernal_sizes=(5, 9, 13),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 norm_eval=False,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super().__init__(init_cfg)
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('frozen_stages must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        self.widen_factor = widen_factor
        self.deepen_factor = deepen_factor # todo

        self.stem = Focus(
            3,
            int(arch_setting[0][0] * widen_factor[0]),
            kernel_size=3,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.layers = ['stem']

        for i, (in_channels, out_channels, num_blocks, add_identity,
                use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor[i])
            out_channels = int(out_channels * widen_factor[i + 1])
            num_blocks = max(round(num_blocks * deepen_factor[i]), 1)
            stage = []
            conv_layer = conv(
                in_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(conv_layer)
            if use_spp:
                spp = SPPBottleneck(
                    out_channels,
                    out_channels,
                    kernel_sizes=spp_kernal_sizes,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                stage.append(spp)
            csp_layer = CSPLayer(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(csp_layer)
            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(SearchableCSPDarknet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, (_BatchNorm, BatchNorm2d)):
                    m.eval()

    def forward(self, x):
        # print('train backbone')
        outs = []
        arch_setting = self.arch_settings['P5']
        stage = 0
        # [[64, 128, 3, True, False], [128, 256, 9, True, False],
        #  [256, 512, 9, True, False], [512, 1024, 3, False, True]]
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)

            if layer_name == 'stem':
                x = layer(x)
                continue

            _, _, num_blocks, add_identity, use_spp = arch_setting[stage]

            # conv
            # print("layer[0]")
            # print(layer)
            x = layer[0](x)
            # print("fin!")

            # spp
            use_spp_x = 1 if use_spp else 0
            if use_spp:
                x = layer[1](x)

            # csp layer todo:如何直接调用csp layer的forward函数
            num_blocks = max(round(num_blocks * self.deepen_factor[i - 1]), 1)
            # print("num_blocks"+str(num_blocks))
            x_short = layer[1 + use_spp_x].short_conv(x) # todo name
            x_main = layer[1 + use_spp_x].main_conv(x)

            darknetbottleneck = layer[1 + use_spp_x].blocks  # Sequential
            for block_num in range(num_blocks):
                identity = x_main
                out = darknetbottleneck[block_num].conv1(x_main)
                out = darknetbottleneck[block_num].conv2(out)

                if add_identity:  # 是否有shorcut
                    out = out + identity

            x = torch.cat((x_main, x_short), dim=1)
            x = layer[1 + use_spp_x].final_conv(x)
            # print("csp_fin!")
            if i in self.out_indices:
                outs.append(x)
            stage = stage + 1

        return tuple(outs)

    def set_arch(self, arch, **kwargs):
        # base_channel = 64
        # base_channel = arch['base_c']  # 修改base channel
        # factor = [1, 2, 4 ,8, 16] # 每个stage的in_channel对应basechannel的倍数
        widen_factor = arch['widen_factor_backbone']
        # base_channel = max(int(self.arch_settings['P5'][0][0] * widen_factor // 16 * 16), 16)#todo
        # print("base_channel:"+str(base_channel))
        deepen_factor = arch['deepen_factor']
        self.widen_factor = widen_factor
        # deepen_factor = 1
        self.deepen_factor = deepen_factor
        arch_setting = self.arch_settings['P5']
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            if layer_name == "stem":
                channel = int(arch_setting[0][0] * widen_factor[0] // 16 * 16)
                # todo
                if channel == 0:
                    channel = 16
                layer.conv.conv.out_channels = channel
                layer.conv.bn.num_features = channel
                continue

            in_channels, out_channels, num_blocks, _, use_spp = arch_setting[i - 1] # todo 改成从arch setting里取出in/out channel * widen——factor
            num_blocks = max(round(num_blocks * deepen_factor[i - 1]), 1) # deepfactor
            use_spp_x = 1 if use_spp else 0
            in_channel = int(in_channels * widen_factor[i - 1] // 16 * 16)
            out_channel = int(out_channels * widen_factor[i] // 16 * 16)
            # todo 太丑了
            if in_channel == 0:
                in_channel = 16
            if out_channel == 0:
                out_channel = 16
            # convmodule
            # in_channel = base_channel * factor[i - 1]
            # out_channel = base_channel * factor[i]
            layer[0].conv.in_channels, layer[0].conv.out_channels = in_channel, out_channel
            layer[0].bn.num_features = out_channel
            if use_spp:
                # sppbottleneck  ?conv2_channels = mid_channels * (len(kernel_sizes) + 1)
                in_channel = out_channel
                mid_channel = in_channel // 2
                out_channel = out_channel
                layer[1].conv1.conv.in_channels, layer[1].conv1.conv.out_channels = in_channel, mid_channel
                layer[1].conv1.bn.num_features = mid_channel
                layer[1].conv2.conv.in_channels, layer[1].conv2.conv.out_channels = mid_channel * 4, out_channel
                layer[1].conv2.bn.num_features = out_channel
            # CSPlayer mid_channels = int(out_channels * expand_ratio)
            # num_blocks = max(round(num_blocks * deepen_factor), 1)
            in_channel = out_channel
            mid_channel = int(out_channel * 0.5)
            layer[1 + use_spp_x].main_conv.conv.in_channels, layer[1 + use_spp_x].main_conv.conv.out_channels = in_channel, mid_channel
            layer[1 + use_spp_x].main_conv.bn.num_features = mid_channel
            layer[1 + use_spp_x].short_conv.conv.in_channels, layer[1 + use_spp_x].short_conv.conv.out_channels = in_channel, mid_channel
            layer[1 + use_spp_x].short_conv.bn.num_features = mid_channel
            layer[1 + use_spp_x].final_conv.conv.in_channels, layer[1 + use_spp_x].final_conv.conv.out_channels = mid_channel * 2, out_channel
            layer[1 + use_spp_x].final_conv.bn.num_features = out_channel
            # DarknetBottleneck
            darknetbottleneck = layer[1 + use_spp_x].blocks  # Sequential
            for block_num in range(num_blocks):
                hidden_channel = mid_channel
                darknetbottleneck[block_num].conv1.conv.in_channels, darknetbottleneck[block_num].conv1.conv.out_channels = mid_channel, hidden_channel
                darknetbottleneck[block_num].conv1.bn.num_features = hidden_channel
                darknetbottleneck[block_num].conv2.conv.in_channels, darknetbottleneck[block_num].conv2.conv.out_channels = hidden_channel, mid_channel
                darknetbottleneck[block_num].conv2.bn.num_features = mid_channel

        # print("<<<<<<<<<<<<<<<<<<<<<<")
        # print("the model")
        # for i, layer_name in enumerate(self.layers):
        #     layer = getattr(self, layer_name)
        #     print("i=" + str(i) + ":" + str(layer))

        '''
            # layer.conv.conv.in_channels = 16
            arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 9, True, False],
               [256, 512, 9, True, False], [512, 1024, 3, False, True]]
    }
    in_channels, out_channels, num_blocks, add_identity,
                use_spp
    factor = [1, 2, 4 ,8, 16]'''
