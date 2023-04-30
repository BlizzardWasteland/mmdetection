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


class LSQQuantizerInitPTQ(nn.Module):
    def __init__(self, quant_bit, scale_num=1, quant_on_weight=False):
        assert scale_num == 1 or quant_on_weight == True, "Channel_wise only can be used on weight quantization."
        super(LSQQuantizerInitPTQ, self).__init__()
        self.quant_bit = quant_bit
        self.scale_num = scale_num
        self.quant_on_weight = quant_on_weight

        if self.quant_on_weight:
            self.min_q = -1 * 2 ** (self.quant_bit - 1)
            self.max_q = 2 ** (self.quant_bit - 1) - 1
        else:
            self.min_q = 0
            self.max_q = 2 ** self.quant_bit - 1
        
        self.register_parameter('scale', nn.Parameter(torch.ones(scale_num)))

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
        g = 1.0 / math.sqrt(inp.numel() * self.max_q)
        if self.quant_on_weight and inp.shape[0] < scale.numel():
            cur_scale = grad_scale(scale[:inp.shape[0]], g).view([inp.shape[0]] + [1] * (inp.dim()-1))
        else:
            cur_scale = grad_scale(scale, g).view([self.scale_num] + [1] * (inp.dim()-1))
        out = round_pass(inp / cur_scale).clamp(self.min_q, self.max_q) * cur_scale
        return out

    def calc_qparams(self):
        if self.quant_on_weight:
            scale = self.max_val / self.max_q
        else:
            scale = self.max_val / (self.max_q - self.min_q)
        self.scale.data.copy_(scale)

    def extra_repr(self):
        s = 'quant_bit={0}, scale_num={1}, quant_on_weight={2}'.format(
            self.quant_bit, self.scale_num, self.quant_on_weight)
        return super(LSQQuantizerInitPTQ, self).extra_repr() + s


class LSQQuantizer(nn.Module):
    def __init__(self, quant_bit, scale_num=1, quant_on_weight=False):
        assert scale_num == 1 or quant_on_weight == True, "Channel_wise only can be used on weight quantization."
        super(LSQQuantizer, self).__init__()
        self.quant_bit = quant_bit
        self.scale_num = scale_num
        self.quant_on_weight = quant_on_weight

        if self.quant_on_weight:
            self.min_q = -1 * 2 ** (self.quant_bit - 1)
            self.max_q = 2 ** (self.quant_bit - 1) - 1
        else:
            self.min_q = 0
            self.max_q = 2 ** self.quant_bit - 1

        self.register_parameter('scale', nn.Parameter(torch.ones(scale_num)))
        self.register_buffer('init_state', torch.zeros(1))

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
        if self.training and self.init_state == 0:
            self.init_state.fill_(1)
            inp_detach = inp.detach()
            if self.scale_num == 1:
                inp_abs_mean = inp_detach.abs().mean()
            else:
                assert self.quant_on_weight
                assert inp.shape[0] == self.scale.numel()
                dim = [i for i in range(1, inp_detach.dim())]
                inp_abs_mean = inp_detach.abs().mean(dim=dim)
            scale = 2 * inp_abs_mean / math.sqrt(self.max_q)
            self.scale.data.copy_(scale)
        self.scale.data.clamp_(self.eps)

        g = 1.0 / math.sqrt(inp.numel() * self.max_q)
        if self.quant_on_weight and inp.shape[0] < self.scale.numel():
            cur_scale = grad_scale(self.scale[:inp.shape[0]], g).view([inp.shape[0]] + [1] * (inp.dim()-1))
        else:
            cur_scale = grad_scale(self.scale, g).view([self.scale_num] + [1] * (inp.dim()-1))
        out = round_pass(inp / cur_scale).clamp(self.min_q, self.max_q) * cur_scale
        return out

    def extra_repr(self):
        s = 'quant_bit={0}, scale_num={1}, quant_on_weight={2}'.format(
            self.quant_bit, self.scale_num, self.quant_on_weight)
        return super(LSQQuantizer, self).extra_repr() + s



# class LSQQuantizerV2(nn.Module):
#     def __init__(self, quant_bit, scale_num=1, quant_on_weight=False):
#         assert scale_num == 1 or quant_on_weight == True, "Channel_wise only can be used on weight quantization."
#         super(LSQQuantizerV2, self).__init__()
#         self.quant_bit = quant_bit
#         self.scale_num = scale_num
#         self.quant_on_weight = quant_on_weight

#         if self.quant_on_weight:
#             self.min_q = -1 * 2 ** (self.quant_bit - 1)
#             self.max_q = 2 ** (self.quant_bit - 1) - 1
#         else:
#             self.min_q = 0
#             self.max_q = 2 ** self.quant_bit - 1
        
#         self.register_parameter('scale', nn.Parameter(torch.ones(scale_num)))
#         self.register_buffer('init_state', torch.zeros(1))

#         self.eps = 1e-6

#     def forward(self, inp):
#         assert self.quant_on_weight or inp.min() >= 0
#         if self.training and self.init_state == 0:
#             self.init_state.fill_(1)
#             inp_detach = inp.detach()
#             if self.quant_on_weight:
#                 if self.scale_num == 1:
#                     inp_mean = inp_detach.mean()
#                     inp_std = inp_detach.std()
#                 else:
#                     dim = [i for i in range(1, inp_detach.dim())]
#                     inp_mean = inp_detach.mean(dim=dim)
#                     inp_std = inp_detach.std(dim=dim)
#                 v1 = torch.abs(inp_mean - 3 * inp_std)
#                 v2 = torch.abs(inp_mean + 3 * inp_std)
#                 scale = torch.max(v1, v2) / 2 ** (self.quant_bit - 1)
#             else:
#                 inp_max = inp_detach.max()
#                 inp_min = inp_detach.min()
#                 best_score = 1e+10
#                 for i in range(80):
#                     cur_ub = inp_max * (1.0 - (i * 0.01))
#                     cur_scale = cur_ub / (2 ** self.quant_bit - 1)
#                     inp_q = torch.round(inp_detach / cur_scale).clamp(
#                         self.min_q, self.max_q) * cur_scale
#                     score = lp_loss(inp_detach, inp_q, p=2.0, reduction='all')
#                     if score < best_score:
#                         best_score = score
#                         scale = cur_scale
#             self.scale.data.copy_(scale)
#         self.scale.data.clamp_(self.eps)

#         g = 1.0 / math.sqrt(inp.numel() * self.max_q)
#         cur_scale = grad_scale(self.scale, g).view([self.scale_num] + [1] * (inp.dim()-1))
#         out = round_pass(inp / cur_scale).clamp(self.min_q, self.max_q) * cur_scale
#         return out

#     def extra_repr(self):
#         s = 'quant_bit={0}, scale_num={1}, quant_on_weight={2}'.format(
#             self.quant_bit, self.scale_num, self.quant_on_weight)
#         return super(LSQQuantizerV1, self).extra_repr() + s
