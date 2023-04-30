import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import CONV_LAYERS

from ..quantizers import LSQQuantizer, LSQQuantizerInitPTQ, PTQQuantizer


@CONV_LAYERS.register_module('QConv2d')
class QConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        w_bit=8,
        channel_wise=True
    ):
        super(QConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        self.w_bit = w_bit
        self.channel_wise = channel_wise
        scale_num = out_channels if channel_wise else 1
        self.w_quantizer = LSQQuantizerInitPTQ(self.w_bit, scale_num, True)

    def forward(self, x):
        w = self.w_quantizer(self.weight)
        out = F.conv2d(x, w, self.bias, self.stride, self.padding,
                    self.dilation, self.groups)
        return out

    @classmethod
    def build_from_original(cls, m_fp, w_bit, channel_wise):
        m = cls(m_fp.in_channels, m_fp.out_channels, m_fp.kernel_size,
                m_fp.stride, m_fp.padding, m_fp.dilation, m_fp.groups,
                m_fp.bias is not None, m_fp.padding_mode,
                w_bit, channel_wise)
        return m


@CONV_LAYERS.register_module('DyQConv2d')
class DyQConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        w_bit=[8],
        channel_wise=True
    ):
        super(DyQConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        self.w_bit = w_bit
        self.channel_wise = channel_wise
        scale_num = out_channels if channel_wise else 1

        self.bit2idx = {bit: i for i, bit in enumerate(self.w_bit)}
        self.idx2bit = {i: bit for i, bit in enumerate(self.w_bit)}

        self.cur_idx = 0
        self.cur_bit = self.idx2bit[self.cur_idx]

        self.w_quantizers = nn.ModuleList([LSQQuantizer(bit, scale_num, True) for bit in self.w_bit])

    def forward(self, x):
        w = self.w_quantizers[self.cur_idx](self.weight)
        out = F.conv2d(x, w, self.bias, self.stride, self.padding,
                    self.dilation, self.groups)
        return out

    def update_bit(self, bit):
        self.cur_bit = bit
        self.cur_idx = self.bit2idx[bit]

    @classmethod
    def build_from_original(cls, m_fp, w_bit, channel_wise):
        m = cls(m_fp.in_channels, m_fp.out_channels, m_fp.kernel_size,
                m_fp.stride, m_fp.padding, m_fp.dilation, m_fp.groups,
                m_fp.bias is not None, m_fp.padding_mode,
                w_bit, channel_wise)
        return m

    def extra_repr(self):
        return super(DyQConv2d, self).extra_repr() + \
            ', bits={0}, cur_bit={1}'.format(self.w_bit, self.cur_bit)



@CONV_LAYERS.register_module('QConv2d_PTQ')
class QConv2d_PTQ(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        w_bit=8,
        channel_wise=True
    ):
        super(QConv2d_PTQ, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        self.w_bit = w_bit
        self.channel_wise = channel_wise
        scale_num = out_channels if channel_wise else 1
        self.w_quantizer = PTQQuantizer(self.w_bit, scale_num, True)


    def forward(self, x):
        w = self.w_quantizer(self.weight)
        out = F.conv2d(x, w, self.bias, self.stride, self.padding,
                    self.dilation, self.groups)
        return out

    @classmethod
    def build_from_original(cls, m_fp, w_bit, channel_wise):
        m = cls(m_fp.in_channels, m_fp.out_channels, m_fp.kernel_size,
                m_fp.stride, m_fp.padding, m_fp.dilation, m_fp.groups,
                m_fp.bias is not None, m_fp.padding_mode,
                w_bit, channel_wise)
        return m



@CONV_LAYERS.register_module('DyQConv2d_PTQ')
class DyQConv2d_PTQ(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        w_bit=[8],
        channel_wise=True
    ):
        super(DyQConv2d_PTQ, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        self.w_bit = w_bit
        self.channel_wise = channel_wise
        scale_num = out_channels if channel_wise else 1

        self.bit2idx = {bit: i for i, bit in enumerate(self.w_bit)}
        self.idx2bit = {i: bit for i, bit in enumerate(self.w_bit)}

        self.cur_idx = 0
        self.cur_bit = self.idx2bit[self.cur_idx]

        self.w_quantizers = nn.ModuleList([PTQQuantizer(bit, scale_num, True) for bit in self.w_bit])

    def forward(self, x):
        w = self.w_quantizers[self.cur_idx](self.weight)
        out = F.conv2d(x, w, self.bias, self.stride, self.padding,
                    self.dilation, self.groups)

    def update_bit(self, bit):
        self.cur_bit = bit
        self.cur_idx = self.bit2idx[bit]

    @classmethod
    def build_from_original(cls, m_fp, w_bit, channel_wise):
        m = cls(m_fp.in_channels, m_fp.out_channels, m_fp.kernel_size,
                m_fp.stride, m_fp.padding, m_fp.dilation, m_fp.groups,
                m_fp.bias is not None, m_fp.padding_mode,
                w_bit, channel_wise)
        return m

    def forward(self, x):
        w = self.w_quantizers[self.cur_idx](self.weight)
        out = F.conv2d(x, w, self.bias, self.stride, self.padding,
                    self.dilation, self.groups)
        return out

    def extra_repr(self):
        return super(DyQConv2d_PTQ, self).extra_repr() + \
            ', bits={0}, cur_bit={1}'.format(self.w_bit, self.cur_bit)


# class QLinear(nn.Linear):
#     def __init__(
#         self,
#         w_bit,
#         a_bit,
#         in_features,
#         out_features,
#         bias=True,
#         channel_wise=False
#     ):
#         super(QLinear, self).__init__(in_features, out_features, bias)
#         self.w_bit = w_bit
#         self.a_bit = a_bit
#         self.channel_wise = channel_wise
#         scale_num = out_features if channel_wise else 1
#         self.w_quantizer = LSQQuantizerV1(self.w_bit, scale_num, True)
#         self.a_quantizer = LSQQuantizerV1(self.a_bit, 1, False)


#     def forward(self, x):
#         x = self.a_quantizer(x)
#         w = self.w_quantizer(self.weight)
#         out = F.linear(x, w, self.bias)
#         return out
