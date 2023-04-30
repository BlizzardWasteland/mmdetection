# Copyright (c) OpenMMLab. All rights reserved.
import random
import torch
from mmcv.runner import get_dist_info, load_state_dict

from mmdet.models.builder import DETECTORS
from .yolox_kd import YOLOX_KD
from ..algorithm.searchbase import SearchBase

from ..utils import DyQConv2d


def map_params(param, tgt_shape):
    new_param = torch.zeros(tgt_shape)
    if param.shape == tgt_shape:
        new_param.copy_(param)
        return new_param
    elif len(param.shape) == 4: # conv weight
        oc, ic, k, _ = param.shape
        oc_t, ic_t, k_t, _ = tgt_shape
        oc_min = min(oc, oc_t)
        ic_min = min(ic, ic_t)
        k_min = min(k, k_t)
        new_param.narrow(0, 0, oc_min).narrow(1, 0, ic_min).narrow(
            2, (k_t - k_min) // 2, k_min).narrow(3, (k_t - k_min) // 2, k_min).copy_(
            param.narrow(0, 0, oc_min).narrow(1, 0, ic_min).narrow(
            2, (k - k_min) // 2, k_min).narrow(3, (k - k_min) // 2, k_min))
        return new_param
    elif len(param.shape) == 1: # bn
        channel = param.shape[0]
        channel_t = tgt_shape[0]
        channel_min = min(channel, channel_t)
        new_param.narrow(0, 0, channel_min).copy_(param.narrow(0, 0, channel_min))
        return new_param
    else:
        raise NotImplementedError


@DETECTORS.register_module()
class SearchableYOLOX_KD(SearchBase, YOLOX_KD):
    def __init__(self,
                 *args,
                 search_space=None,
                 conv2d_quant_bit=None,
                 bn_training_mode=True,
                 num_sample_training=4,
                 divisor=4,
                 retraining=False,
                 **kwargs
                 ):
        YOLOX_KD.__init__(self, *args, **kwargs)
        SearchBase.__init__(self, bn_training_mode=bn_training_mode, num_sample_training=num_sample_training, divisor=divisor, retraining=retraining)
        self._random_size_interval = self._random_size_interval*self.num_sample_training

        self.search_space = search_space
        if self.search_space:
            self.backbone_widen_factor_range = search_space['backbone_widen_factor_range']
            self.backbone_deepen_factor_range = search_space['backbone_deepen_factor_range']
            self.neck_widen_factor_range = search_space['neck_widen_factor_range']
            self.head_widen_factor_range = search_space['head_widen_factor_range']
            if 'conv2d_quant_bits' in search_space:
                self.conv2d_quant_bits = search_space['conv2d_quant_bits']

        if conv2d_quant_bit is not None:
            cnt = 0
            for m in self.modules():
                if isinstance(m, DyQConv2d):
                    m.update_bit(conv2d_quant_bit[cnt])
                    cnt += 1

    def sample_arch(self, mode='random'):
        assert mode in ('max', 'min', 'random')
        arch = {}
        if mode in ('max', 'min'):
            fn = eval(mode)
            arch['widen_factor_backbone'] = tuple([fn(self.backbone_widen_factor_range)]*5)
            arch['deepen_factor_backbone'] = tuple([fn(self.backbone_deepen_factor_range)]*4)
            arch['widen_factor_neck'] = tuple([fn(self.neck_widen_factor_range)]*8)
            arch['widen_factor_neck_out'] = fn(self.head_widen_factor_range)
            if 'conv2d_quant_bits' in self.search_space:
                quant_conv2d_cnt = [isinstance(m, DyQConv2d) for m in self.modules()].count(True)
                arch['conv2d_quant_bit'] = tuple([fn(self.conv2d_quant_bits)]*quant_conv2d_cnt)
        elif mode == 'random':
            arch['widen_factor_backbone'] = tuple(random.choices(self.backbone_widen_factor_range, k=5))
            arch['deepen_factor_backbone'] = tuple(random.choices(self.backbone_deepen_factor_range, k=4))
            arch['widen_factor_neck'] = tuple(random.choices(self.neck_widen_factor_range, k=8))
            arch['widen_factor_neck_out'] = random.choice(self.head_widen_factor_range)
            if 'conv2d_quant_bits' in self.search_space:
                quant_conv2d_cnt = [isinstance(m, DyQConv2d) for m in self.modules()].count(True)
                arch['conv2d_quant_bit'] = tuple(random.choices(self.conv2d_quant_bits, k=quant_conv2d_cnt))
        else:
            raise NotImplementedError
        return arch

    def set_arch(self, arch_dict):
        if 'conv2d_quant_bits' in self.search_space:
            cnt = 0
            for m in self.modules():
                if isinstance(m, DyQConv2d):
                    m.update_bit(arch_dict['conv2d_quant_bit'][cnt])
                    cnt += 1
        self.backbone.set_arch(arch_dict, divisor=self.divisor)
        self.neck.set_arch(arch_dict, divisor=self.divisor)
        self.bbox_head.set_arch(arch_dict, divisor=self.divisor)

    def init_weights(self):
        if self.init_cfg is None:
            YOLOX_KD.init_weights(self)
            return
        from mmcv.runner import CheckpointLoader
        state_dict = CheckpointLoader.load_checkpoint(self.init_cfg['checkpoint'], 'cpu')['state_dict']
        cur_state_dict = self.state_dict()
        new_state_dict = {}
        for k, v in cur_state_dict.items():
            if k in state_dict:
                new_state_dict[k] = map_params(state_dict[k], v.shape)
        load_state_dict(self, new_state_dict)
