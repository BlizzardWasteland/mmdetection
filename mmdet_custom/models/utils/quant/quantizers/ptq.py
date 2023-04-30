import torch
import torch.nn as nn
import math


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


class PTQQuantizer(nn.Module):
    def __init__(self, quant_bit, scale_num=1, quant_on_weight=False):
        assert scale_num == 1 or quant_on_weight == True, "Channel_wise only can be used on weight quantization."
        super(PTQQuantizer, self).__init__()
        self.quant_bit = quant_bit
        self.scale_num = scale_num
        self.quant_on_weight = quant_on_weight

        if self.quant_on_weight:
            self.min_q = -1 * 2 ** (self.quant_bit - 1)
            self.max_q = 2 ** (self.quant_bit - 1) - 1
        else:
            self.min_q = 0
            self.max_q = 2 ** self.quant_bit - 1

        self.register_buffer('scale', torch.ones(scale_num))

        self.max_val = None

        self.calib = False
        self.eps = 1e-6

    def forward(self, inp):
        if torch.onnx.is_in_onnx_export():
            scale = self.scale
            if self.scale_num != 1:
                assert self.quant_on_weight
                scale = scale.view([inp.shape[0]] + [1] * (inp.dim()-1))
            out = torch.round(inp / scale).clamp(self.min_q, self.max_q) * scale
            return out
        assert self.quant_on_weight or inp.min() >= 0
        if self.calib:
            inp_detach = inp.detach()
            if self.scale_num == 1:
                inp_abs_max = inp_detach.abs().max()
            else:
                assert self.quant_on_weight
                assert inp.shape[0] == self.scale.numel()
                inp_abs_max = inp_detach.abs().view(inp.shape[0], -1).max(dim=-1).values
            if self.max_val is None:
                self.max_val = inp_abs_max
            else:
                self.max_val = torch.maximum(self.max_val, inp_abs_max)
            if self.quant_on_weight:
                scale = self.max_val / self.max_q
            else:
                scale = self.max_val / (self.max_q - self.min_q)
            scale.clamp_(self.eps)
        else:
            self.scale.data.clamp_(self.eps)
            scale = self.scale

        if self.quant_on_weight and inp.shape[0] < self.scale.numel():
            cur_scale = scale[:inp.shape[0]].view([inp.shape[0]] + [1] * (inp.dim()-1))
        else:
            cur_scale = scale.view([self.scale_num] + [1] * (inp.dim()-1))
        out = torch.round(inp / cur_scale).clamp(self.min_q, self.max_q) * cur_scale
        return out

    def calc_qparams(self):
        if self.max_val is not None:
            if self.quant_on_weight:
                self.scale.data.copy_(self.max_val / self.max_q)
            else:
                self.scale.data.copy_(self.max_val / (self.max_q - self.min_q))

    def extra_repr(self):
        s = 'quant_bit={0}, scale_num={1}, quant_on_weight={2}'.format(
            self.quant_bit, self.scale_num, self.quant_on_weight)
        return super(PTQQuantizer, self).extra_repr() + s
