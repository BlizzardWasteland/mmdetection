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

# @NECKS.register_module()
# class SearchableYOLOXPAFPN(YOLOXPAFPN):
#     def set_arch(self, arch, divisor=8):
#         widen_factor_backbone = arch['widen_factor_backbone'][-len(self.in_channels):]
#         in_channels = [make_divisible(c*alpha, divisor) for c,alpha in zip(self.in_channels, widen_factor_backbone)]
#         out_channels = make_divisible(self.out_channels*arch['widen_factor_neck_out'], divisor)
#         for idx in range(len(in_channels) - 1, 0, -1):
#             _idx = len(in_channels)-1-idx
#             set_channels(self.reduce_layers[_idx], 
#                     in_channels[idx], in_channels[idx - 1])
#             set_channel_ratio(self.top_down_blocks[_idx], 
#                     widen_factor_backbone[idx - 1], divisor=divisor)
#             set_channels(self.top_down_blocks[_idx].main_conv.conv, 
#                     in_channels=in_channels[idx - 1] * 2)
#             set_channels(self.top_down_blocks[_idx].short_conv.conv, 
#                     in_channels=in_channels[idx - 1] * 2)

#         for idx in range(len(in_channels) - 1):
#             set_channels(self.downsamples[idx], 
#                     in_channels[idx], in_channels[idx])
#             set_channel_ratio(self.bottom_up_blocks[idx], 
#                     widen_factor_backbone[idx+1], divisor=divisor)
#             set_channels(self.bottom_up_blocks[idx].main_conv.conv, 
#                     in_channels=in_channels[idx] * 2)
#             set_channels(self.bottom_up_blocks[idx].short_conv.conv, 
#                     in_channels=in_channels[idx] * 2)
        
#         for i in range(len(in_channels)):
#             set_channels(self.out_convs, in_channels[i], out_channels)
#         print(self)
#         print(in_channels)

@NECKS.register_module()
class SearchableYOLOXPAFPN(YOLOXPAFPN):
    """Path Aggregation Network used in YOLOX.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 3
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish')
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels=[256, 512, 1024],
                 base_in_channels=None,
                 out_channels=256,
                 widen_factor=[1]*8,
                 num_csp_blocks=3,
                 use_depthwise=False,
                 upsample_cfg=dict(scale_factor=2, mode='nearest'),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        BaseModule.__init__(self, init_cfg)
        self.widen_factor = widen_factor
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()

        # self.base_channels_backbone = [256, 512, 1024]
        # base_channels_dict = {
        #     'reduce_layers0':512,
        #     'reduce_layers1':256,
        #     'top_down_blocks0':512,
        #     'top_down_blocks1':256,
        #     'downsamples0':256,
        #     'downsamples1':512,
        #     'bottom_up_blocks0':512,
        #     'bottom_up_blocks1':1024
        # }
        self.base_in_channels = in_channels if base_in_channels is None else base_in_channels
        base_channels_dict = {
            'reduce_layers0':self.base_in_channels[1],
            'reduce_layers1':self.base_in_channels[0],
            'top_down_blocks0':self.base_in_channels[1],
            'top_down_blocks1':self.base_in_channels[0],
            'downsamples0':self.base_in_channels[0],
            'downsamples1':self.base_in_channels[1],
            'bottom_up_blocks0':self.base_in_channels[1],
            'bottom_up_blocks1':self.base_in_channels[2]
        }
        self.base_channels_dict = base_channels_dict
        
        # create factor_dictionary
        widen_factor_dict = {
            'reduce_layers0':widen_factor[0],
            'reduce_layers1':widen_factor[1],
            'top_down_blocks0':widen_factor[2],
            'top_down_blocks1':widen_factor[3],
            'downsamples0':widen_factor[4],
            'downsamples1':widen_factor[5],
            'bottom_up_blocks0':widen_factor[6],
            'bottom_up_blocks1':widen_factor[7],
        }

        channels_out_dict = {
            'reduce_layers0':int(widen_factor_dict['reduce_layers0'] * base_channels_dict['reduce_layers0']),
            'reduce_layers1': int(widen_factor_dict['reduce_layers1'] * base_channels_dict['reduce_layers1']),
            'top_down_blocks0': int(widen_factor_dict['top_down_blocks0'] * base_channels_dict['top_down_blocks0']),
            'top_down_blocks1': int(widen_factor_dict['top_down_blocks1'] * base_channels_dict['top_down_blocks1']),
            'downsamples0': int(widen_factor_dict['downsamples0'] * base_channels_dict['downsamples0']),
            'downsamples1': int(widen_factor_dict['downsamples1'] * base_channels_dict['downsamples1']),
            'bottom_up_blocks0': int(widen_factor_dict['bottom_up_blocks0'] * base_channels_dict['bottom_up_blocks0']),
            'bottom_up_blocks1': int(widen_factor_dict['bottom_up_blocks1'] * base_channels_dict['bottom_up_blocks1']),
        }

        channels_dict = {
            'reduce_layers0': [in_channels[2], channels_out_dict['reduce_layers0']],
            'reduce_layers1': [channels_out_dict['top_down_blocks0'], channels_out_dict['reduce_layers1']],
            'top_down_blocks0': [(in_channels[1] + channels_out_dict['reduce_layers0']), channels_out_dict['top_down_blocks0']],
            'top_down_blocks1': [(in_channels[0] + channels_out_dict['reduce_layers1']), channels_out_dict['top_down_blocks1']],
            'downsamples0': [channels_out_dict['top_down_blocks1'], channels_out_dict['downsamples0']],
            'downsamples1': [channels_out_dict['bottom_up_blocks0'], channels_out_dict['downsamples1']],
            'bottom_up_blocks0': [(channels_out_dict['reduce_layers1'] + channels_out_dict['downsamples0']), channels_out_dict['bottom_up_blocks0']],
            'bottom_up_blocks1': [(channels_out_dict['reduce_layers0'] + channels_out_dict['downsamples1']), channels_out_dict['bottom_up_blocks1']],
        }

        # build top-down blocks
        for idx in range(len(in_channels) - 1, 0, -1):
            layer_name_reduce = 'reduce_layers'+str(len(in_channels) -1 - idx)
            layer_name_td = 'top_down_blocks'+str(len(in_channels) -1 - idx)
            self.reduce_layers.append(
                ConvModule(
                    channels_dict[layer_name_reduce][0],
                    channels_dict[layer_name_reduce][1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.top_down_blocks.append(
                CSPLayer(
                    channels_dict[layer_name_td][0],
                    channels_dict[layer_name_td][1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            layer_name_downsample = 'downsamples' + str(idx)
            layer_name_bu = 'bottom_up_blocks' + str(idx)

            self.downsamples.append(
                conv(
                    channels_dict[layer_name_downsample][0],
                    channels_dict[layer_name_downsample][1],
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            # print("bottom_up_blocks")
            # print(new_channels_reduce[1 - idx] + new_channels_bu[idx])
            self.bottom_up_blocks.append(
                CSPLayer(
                    channels_dict[layer_name_bu][0],
                    channels_dict[layer_name_bu][1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.out_convs = nn.ModuleList()
        out_convs_in_channel = [channels_dict['top_down_blocks1'][1],
                             channels_dict['bottom_up_blocks0'][1],
                             channels_dict['bottom_up_blocks1'][1]]
        for i in range(len(in_channels)):
            self.out_convs.append(
                ConvModule(
                    out_convs_in_channel[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        
    def set_arch(self, arch, **kwargs):
        widen_factor_backbone = arch['widen_factor_backbone'][-len(self.in_channels):]
        in_channels = []
        # for c, alpha in zip([256, 512, 1024], widen_factor_backbone):
        #     in_channels.append(int(c*alpha))
        for c, alpha in zip(self.in_channels, widen_factor_backbone):
            in_channels.append(int(c*alpha))

        # out_channels = int(256 * widen_factor_out_neck)
        out_channels = int(self.out_channels * arch['widen_factor_neck_out'])

        widen_factor = arch['widen_factor_neck']
        widen_factor_dict = {
            'reduce_layers0':widen_factor[0],
            'reduce_layers1':widen_factor[1],
            'top_down_blocks0':widen_factor[2],
            'top_down_blocks1':widen_factor[3],
            'downsamples0':widen_factor[4],
            'downsamples1':widen_factor[5],
            'bottom_up_blocks0':widen_factor[6],
            'bottom_up_blocks1':widen_factor[7],
        }
        base_channels_dict = {
            'reduce_layers0':in_channels[1],
            'reduce_layers1':in_channels[0],
            'top_down_blocks0':in_channels[1],
            'top_down_blocks1':in_channels[0],
            'downsamples0':in_channels[0],
            'downsamples1':in_channels[1],
            'bottom_up_blocks0':in_channels[1],
            'bottom_up_blocks1':in_channels[2]
        }
        channels_out_dict = {
            'reduce_layers0':int(widen_factor_dict['reduce_layers0'] * base_channels_dict['reduce_layers0']),
            'reduce_layers1': int(widen_factor_dict['reduce_layers1'] * base_channels_dict['reduce_layers1']),
            'top_down_blocks0': int(widen_factor_dict['top_down_blocks0'] * base_channels_dict['top_down_blocks0']),
            'top_down_blocks1': int(widen_factor_dict['top_down_blocks1'] * base_channels_dict['top_down_blocks1']),
            'downsamples0': int(widen_factor_dict['downsamples0'] * base_channels_dict['downsamples0']),
            'downsamples1': int(widen_factor_dict['downsamples1'] * base_channels_dict['downsamples1']),
            'bottom_up_blocks0': int(widen_factor_dict['bottom_up_blocks0'] * base_channels_dict['bottom_up_blocks0']),
            'bottom_up_blocks1': int(widen_factor_dict['bottom_up_blocks1'] * base_channels_dict['bottom_up_blocks1']),
        }
        channels_dict = {
            'reduce_layers0': [in_channels[2], channels_out_dict['reduce_layers0']],
            'reduce_layers1': [channels_out_dict['top_down_blocks0'], channels_out_dict['reduce_layers1']],
            'top_down_blocks0': [(in_channels[1] + channels_out_dict['reduce_layers0']), channels_out_dict['top_down_blocks0']],
            'top_down_blocks1': [(in_channels[0] + channels_out_dict['reduce_layers1']), channels_out_dict['top_down_blocks1']],
            'downsamples0': [channels_out_dict['top_down_blocks1'], channels_out_dict['downsamples0']],
            'downsamples1': [channels_out_dict['bottom_up_blocks0'], channels_out_dict['downsamples1']],
            'bottom_up_blocks0': [(channels_out_dict['reduce_layers1'] + channels_out_dict['downsamples0']), channels_out_dict['bottom_up_blocks0']],
            'bottom_up_blocks1': [(channels_out_dict['reduce_layers0'] + channels_out_dict['downsamples1']), channels_out_dict['bottom_up_blocks1']],
        }

        expansion_ratio = 0.5 # todo 搞清楚这是个啥

        for idx in range(len(self.in_channels) - 1): # 0, 1
            # reduce_layers
            layer_name_reduce = 'reduce_layers' + str(idx)
            layer_name_td = 'top_down_blocks' + str(idx)
            self.reduce_layers[idx].conv.in_channels = channels_dict[layer_name_reduce][0]
            self.reduce_layers[idx].conv.out_channels = channels_dict[layer_name_reduce][1]
            self.reduce_layers[idx].bn.num_features = channels_dict[layer_name_reduce][1]
            # top_down_blocks
            mid_channel = int(channels_dict[layer_name_td][1] * expansion_ratio)
            self.top_down_blocks[idx].main_conv.conv.in_channels = channels_dict[layer_name_td][0]
            self.top_down_blocks[idx].main_conv.conv.out_channels = mid_channel
            self.top_down_blocks[idx].main_conv.bn.num_features = mid_channel
            self.top_down_blocks[idx].short_conv.conv.in_channels = channels_dict[layer_name_td][0]
            self.top_down_blocks[idx].short_conv.conv.out_channels = mid_channel
            self.top_down_blocks[idx].short_conv.bn.num_features = mid_channel
            self.top_down_blocks[idx].final_conv.conv.in_channels = 2 * mid_channel
            self.top_down_blocks[idx].final_conv.conv.out_channels = channels_dict[layer_name_td][1]
            self.top_down_blocks[idx].final_conv.bn.num_features = channels_dict[layer_name_td][1]

            for block in self.top_down_blocks[idx].blocks:
                hidden_channel = mid_channel
                block.conv1.conv.in_channels, block.conv1.conv.out_channels = mid_channel, hidden_channel
                block.conv1.bn.num_features = hidden_channel
                block.conv2.conv.in_channels, block.conv2.conv.out_channels = hidden_channel, mid_channel
                block.conv2.bn.num_features = mid_channel

        for idx in range(len(self.in_channels) - 1): # 0, 1
            layer_name_downsample = 'downsamples' + str(idx)
            layer_name_bu = 'bottom_up_blocks' + str(idx)
            # downsamples
            self.downsamples[idx].conv.in_channels = channels_dict[layer_name_downsample][0]
            self.downsamples[idx].conv.out_channels = channels_dict[layer_name_downsample][1]
            self.downsamples[idx].bn.num_features = channels_dict[layer_name_downsample][1]
            # bottom_up_blocks
            mid_channel = int(channels_dict[layer_name_bu][1] * expansion_ratio) # 128
            self.bottom_up_blocks[idx].main_conv.conv.in_channels = channels_dict[layer_name_bu][0]
            self.bottom_up_blocks[idx].main_conv.conv.out_channels = mid_channel
            self.bottom_up_blocks[idx].main_conv.bn.num_features = mid_channel
            self.bottom_up_blocks[idx].short_conv.conv.in_channels = channels_dict[layer_name_bu][0]
            self.bottom_up_blocks[idx].short_conv.conv.out_channels = mid_channel
            self.bottom_up_blocks[idx].short_conv.bn.num_features = mid_channel
            self.bottom_up_blocks[idx].final_conv.conv.in_channels = 2 * mid_channel
            self.bottom_up_blocks[idx].final_conv.conv.out_channels = channels_dict[layer_name_bu][1]
            self.bottom_up_blocks[idx].final_conv.bn.num_features = channels_dict[layer_name_bu][1]

            for block in self.bottom_up_blocks[idx].blocks:
                hidden_channel = mid_channel # 128
                block.conv1.conv.in_channels, block.conv1.conv.out_channels = mid_channel, hidden_channel
                block.conv1.bn.num_features = hidden_channel
                block.conv2.conv.in_channels, block.conv2.conv.out_channels = hidden_channel, mid_channel
                block.conv2.bn.num_features = mid_channel

        # out_convs
        out_convs_in_channel = [channels_dict['top_down_blocks1'][1],
                                channels_dict['bottom_up_blocks0'][1],
                                channels_dict['bottom_up_blocks1'][1]]
        for idx in range(len(self.in_channels)):
            out_channel = out_channels #todo 现在都是固定的，以后改成可变的
            self.out_convs[idx].conv.in_channels = out_convs_in_channel[idx]
            self.out_convs[idx].conv.out_channels = out_channel
            self.out_convs[idx].bn.num_features = out_channel


@NECKS.register_module()
class SearchableYOLOXPAFPNv2(SearchableYOLOXPAFPN):
    def set_arch(self, arch, **kwargs):
        widen_factor_backbone = arch['widen_factor_backbone'][-len(self.in_channels):]
        in_channels = []
        # for c, alpha in zip([256, 512, 1024], widen_factor_backbone):
        #     in_channels.append(int(c*alpha))
        for c, alpha in zip(self.in_channels, widen_factor_backbone):
            in_channels.append(int(c*alpha))

        # out_channels = int(256 * widen_factor_out_neck)
        out_channels = int(self.out_channels * arch['widen_factor_neck_out'])

        widen_factor = arch['widen_factor_neck']
        widen_factor_dict = {
            'reduce_layers0':widen_factor[0],
            'reduce_layers1':widen_factor[1],
            'top_down_blocks0':widen_factor[2],
            'top_down_blocks1':widen_factor[3],
            'downsamples0':widen_factor[4],
            'downsamples1':widen_factor[5],
            'bottom_up_blocks0':widen_factor[6],
            'bottom_up_blocks1':widen_factor[7],
        }
        base_channels_dict = self.base_channels_dict
        channels_out_dict = {
            'reduce_layers0':int(widen_factor_dict['reduce_layers0'] * base_channels_dict['reduce_layers0']),
            'reduce_layers1': int(widen_factor_dict['reduce_layers1'] * base_channels_dict['reduce_layers1']),
            'top_down_blocks0': int(widen_factor_dict['top_down_blocks0'] * base_channels_dict['top_down_blocks0']),
            'top_down_blocks1': int(widen_factor_dict['top_down_blocks1'] * base_channels_dict['top_down_blocks1']),
            'downsamples0': int(widen_factor_dict['downsamples0'] * base_channels_dict['downsamples0']),
            'downsamples1': int(widen_factor_dict['downsamples1'] * base_channels_dict['downsamples1']),
            'bottom_up_blocks0': int(widen_factor_dict['bottom_up_blocks0'] * base_channels_dict['bottom_up_blocks0']),
            'bottom_up_blocks1': int(widen_factor_dict['bottom_up_blocks1'] * base_channels_dict['bottom_up_blocks1']),
        }
        channels_dict = {
            'reduce_layers0': [in_channels[2], channels_out_dict['reduce_layers0']],
            'reduce_layers1': [channels_out_dict['top_down_blocks0'], channels_out_dict['reduce_layers1']],
            'top_down_blocks0': [(in_channels[1] + channels_out_dict['reduce_layers0']), channels_out_dict['top_down_blocks0']],
            'top_down_blocks1': [(in_channels[0] + channels_out_dict['reduce_layers1']), channels_out_dict['top_down_blocks1']],
            'downsamples0': [channels_out_dict['top_down_blocks1'], channels_out_dict['downsamples0']],
            'downsamples1': [channels_out_dict['bottom_up_blocks0'], channels_out_dict['downsamples1']],
            'bottom_up_blocks0': [(channels_out_dict['reduce_layers1'] + channels_out_dict['downsamples0']), channels_out_dict['bottom_up_blocks0']],
            'bottom_up_blocks1': [(channels_out_dict['reduce_layers0'] + channels_out_dict['downsamples1']), channels_out_dict['bottom_up_blocks1']],
        }

        expansion_ratio = 0.5 # todo 搞清楚这是个啥

        for idx in range(len(self.in_channels) - 1): # 0, 1
            # reduce_layers
            layer_name_reduce = 'reduce_layers' + str(idx)
            layer_name_td = 'top_down_blocks' + str(idx)
            self.reduce_layers[idx].conv.in_channels = channels_dict[layer_name_reduce][0]
            self.reduce_layers[idx].conv.out_channels = channels_dict[layer_name_reduce][1]
            self.reduce_layers[idx].bn.num_features = channels_dict[layer_name_reduce][1]
            # top_down_blocks
            mid_channel = int(channels_dict[layer_name_td][1] * expansion_ratio)
            self.top_down_blocks[idx].main_conv.conv.in_channels = channels_dict[layer_name_td][0]
            self.top_down_blocks[idx].main_conv.conv.out_channels = mid_channel
            self.top_down_blocks[idx].main_conv.bn.num_features = mid_channel
            self.top_down_blocks[idx].short_conv.conv.in_channels = channels_dict[layer_name_td][0]
            self.top_down_blocks[idx].short_conv.conv.out_channels = mid_channel
            self.top_down_blocks[idx].short_conv.bn.num_features = mid_channel
            self.top_down_blocks[idx].final_conv.conv.in_channels = 2 * mid_channel
            self.top_down_blocks[idx].final_conv.conv.out_channels = channels_dict[layer_name_td][1]
            self.top_down_blocks[idx].final_conv.bn.num_features = channels_dict[layer_name_td][1]

            for block in self.top_down_blocks[idx].blocks:
                hidden_channel = mid_channel
                block.conv1.conv.in_channels, block.conv1.conv.out_channels = mid_channel, hidden_channel
                block.conv1.bn.num_features = hidden_channel
                block.conv2.conv.in_channels, block.conv2.conv.out_channels = hidden_channel, mid_channel
                block.conv2.bn.num_features = mid_channel

        for idx in range(len(self.in_channels) - 1): # 0, 1
            layer_name_downsample = 'downsamples' + str(idx)
            layer_name_bu = 'bottom_up_blocks' + str(idx)
            # downsamples
            self.downsamples[idx].conv.in_channels = channels_dict[layer_name_downsample][0]
            self.downsamples[idx].conv.out_channels = channels_dict[layer_name_downsample][1]
            self.downsamples[idx].bn.num_features = channels_dict[layer_name_downsample][1]
            # bottom_up_blocks
            mid_channel = int(channels_dict[layer_name_bu][1] * expansion_ratio) # 128
            self.bottom_up_blocks[idx].main_conv.conv.in_channels = channels_dict[layer_name_bu][0]
            self.bottom_up_blocks[idx].main_conv.conv.out_channels = mid_channel
            self.bottom_up_blocks[idx].main_conv.bn.num_features = mid_channel
            self.bottom_up_blocks[idx].short_conv.conv.in_channels = channels_dict[layer_name_bu][0]
            self.bottom_up_blocks[idx].short_conv.conv.out_channels = mid_channel
            self.bottom_up_blocks[idx].short_conv.bn.num_features = mid_channel
            self.bottom_up_blocks[idx].final_conv.conv.in_channels = 2 * mid_channel
            self.bottom_up_blocks[idx].final_conv.conv.out_channels = channels_dict[layer_name_bu][1]
            self.bottom_up_blocks[idx].final_conv.bn.num_features = channels_dict[layer_name_bu][1]

            for block in self.bottom_up_blocks[idx].blocks:
                hidden_channel = mid_channel # 128
                block.conv1.conv.in_channels, block.conv1.conv.out_channels = mid_channel, hidden_channel
                block.conv1.bn.num_features = hidden_channel
                block.conv2.conv.in_channels, block.conv2.conv.out_channels = hidden_channel, mid_channel
                block.conv2.bn.num_features = mid_channel

        # out_convs
        out_convs_in_channel = [channels_dict['top_down_blocks1'][1],
                                channels_dict['bottom_up_blocks0'][1],
                                channels_dict['bottom_up_blocks1'][1]]
        for idx in range(len(self.in_channels)):
            out_channel = out_channels #todo 现在都是固定的，以后改成可变的
            self.out_convs[idx].conv.in_channels = out_convs_in_channel[idx]
            self.out_convs[idx].conv.out_channels = out_channel
            self.out_convs[idx].bn.num_features = out_channel
