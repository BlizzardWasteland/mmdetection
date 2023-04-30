# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule

from mmdet.models.builder import NECKS
from mmdet.models.utils import CSPLayer
from mmdet.models.necks import YOLOXPAFPN

from . import SearchableYOLOXPAFPN, SearchableYOLOXPAFPNv2
from ..utils import QuantCSPLayer
from ..utils.usconv import set_channel_ratio, make_divisible, set_channels


@NECKS.register_module()
class QuantSearchableYOLOXPAFPN(SearchableYOLOXPAFPN):
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
        self.top_down_dummy_relus = nn.ModuleList()

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
                QuantCSPLayer(
                    channels_dict[layer_name_td][0],
                    channels_dict[layer_name_td][1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.top_down_dummy_relus.append(nn.ReLU(inplace=True))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        self.bottom_up_dummy_relus = nn.ModuleList()
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
                QuantCSPLayer(
                    channels_dict[layer_name_bu][0],
                    channels_dict[layer_name_bu][1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.bottom_up_dummy_relus.append(nn.ReLU(inplace=True))

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

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: YOLOXPAFPN features.
        """
        assert len(inputs) == len(self.in_channels)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)
            cat_feat = torch.cat([upsample_feat, feat_low], 1)
            cat_feat = self.top_down_dummy_relus[len(self.in_channels) - 1 - idx](cat_feat)
            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](cat_feat)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            cat_feat = torch.cat([downsample_feat, feat_height], 1)
            cat_feat = self.bottom_up_dummy_relus[idx](cat_feat)
            out = self.bottom_up_blocks[idx](cat_feat)
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        return tuple(outs)


@NECKS.register_module()
class QuantSearchableYOLOXPAFPNv2(QuantSearchableYOLOXPAFPN, SearchableYOLOXPAFPNv2):
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
        QuantSearchableYOLOXPAFPN.__init__(
            self, in_channels, base_in_channels, out_channels, widen_factor, num_csp_blocks,
            use_depthwise, upsample_cfg, conv_cfg, norm_cfg, act_cfg, init_cfg
        )
