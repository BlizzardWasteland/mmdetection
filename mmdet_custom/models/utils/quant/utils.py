import numpy as np
import torch
import torch.nn as nn

def extract_names(m, target_class, conv2d_names=[], base_name=''):
    for name, child in m.named_children():
        now_name = base_name + '.' + name if base_name != '' else name
        if type(child) == target_class:
            conv2d_names.append(now_name)
        else:
            extract_names(child, target_class, conv2d_names, now_name)


def setattr_dot(model, name, module):
    name_list = name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    setattr(model, name_list[-1], module)


def getattr_dot(model, name):
    name_list = name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    return getattr(model, name_list[-1])

def count_qconvNd_params(m, x, y):
    kernel_ops = np.prod(list(m.kernel_size)) # Kw x Kh
    bit = m.cur_bit if hasattr(m, 'cur_bit') else m.w_bit
    m.total_params[0] = kernel_ops*m.in_channels*m.out_channels//m.groups // (32 / bit)
