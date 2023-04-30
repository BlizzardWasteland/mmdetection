# Copyright (c) OpenMMLab. All rights reserved.
import random
import torch
import torch.nn as nn
from mmcv.runner import get_dist_info, load_state_dict

from mmdet.models.builder import DETECTORS
from .yolox_searchable_sandwich_incre_self_training_EWC import SearchableYOLOX_KD_Incre_ST_EWC

from collections import OrderedDict
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDistributedDataParallel
import os
import mmcv
from mmdet.models import build_detector
from mmdet.models.detectors.single_stage import SingleStageDetector
# from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

@DETECTORS.register_module()
class SearchableYOLOX_KD_Incre_ST_EWC_Momentum(SearchableYOLOX_KD_Incre_ST_EWC):
    def __init__(self,
                 momentum,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.base_momentum = momentum
        self.momentum = momentum
    
    def momentum_update(self):
        # for param_t, param_s in zip(self.ori_model.parameters(), self.parameters()):
        #     param_t = self.momentum * param_t + (1 - self.momentum) * param_s
        # current_param = self.named_parameters()
        current_param = self.state_dict()

        for name, p in self.ori_model.named_parameters():
            if 'ema' in name:
                continue 
            if 'multi_level_conv_cls' in name:
                # print(p[0])
                p.data = p.data * self.momentum + \
                             current_param[name][:self.ori_num_classes, ...].data * (1. - self.momentum)
            else:
                p.data = p.data * self.momentum + \
                             current_param[name].data * (1. - self.momentum)

