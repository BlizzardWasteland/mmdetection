import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ACTIVATION_LAYERS

from ..quantizers import LSQQuantizerInitPTQ, PTQQuantizer


@ACTIVATION_LAYERS.register_module('QReLU')
class QReLU(nn.Module):
    def __init__(self, a_bit=8, inplace=False):
        super(QReLU, self).__init__()
        self.a_bit = a_bit
        self.inplace = inplace
        self.a_quantizer = LSQQuantizerInitPTQ(self.a_bit, 1, False)

    def forward(self, x):
        x = F.relu(x, inplace=self.inplace)
        x = self.a_quantizer(x)
        return x

    @classmethod
    def build_from_original(cls, m_fp, a_bit):
        m = cls(a_bit, m_fp.inplace)
        return m

@ACTIVATION_LAYERS.register_module('QReLU_PTQ')
class QReLU_PTQ(nn.Module):
    def __init__(self, a_bit=8, inplace=False):
        super(QReLU_PTQ, self).__init__()
        self.a_bit = a_bit
        self.inplace = inplace
        self.a_quantizer = PTQQuantizer(self.a_bit, 1, False)

    def forward(self, x):
        x = F.relu(x, inplace=self.inplace)
        x = self.a_quantizer(x)
        return x

    @classmethod
    def build_from_original(cls, m_fp, a_bit):
        m = cls(a_bit, m_fp.inplace)
        return m


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
