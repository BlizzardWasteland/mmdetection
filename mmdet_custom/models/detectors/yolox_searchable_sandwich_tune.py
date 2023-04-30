# Copyright (c) OpenMMLab. All rights reserved.
import random
import torch
from mmcv.runner import get_dist_info, load_state_dict

from mmdet.models.builder import DETECTORS
from .yolox_searchable_sandwich import SearchableYOLOX_KD

from collections import OrderedDict
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDistributedDataParallel



@DETECTORS.register_module()
class SearchableYOLOX_KD_Tune(SearchableYOLOX_KD):
    def __init__(self,
                 ori_num_classes,
                 ori_checkpoint_file,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.ori_num_classes = ori_num_classes
        self.load_checkpoint_for_new_model(ori_checkpoint_file, 'cpu')

    def load_checkpoint_for_new_model(self, checkpoint_file, map_location=None, strict=False, logger=None):
        # load ckpt
        checkpoint = torch.load(checkpoint_file, map_location=map_location)
        # get state_dict from checkpoint
        if isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            raise RuntimeError(
                'No state_dict found in checkpoint file {}'.format(checkpoint_file))
        # strip prefix of state_dict
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k,
                          v in checkpoint['state_dict'].items()}
        # modify cls head size of state_dict
        for i in range(len(self.bbox_head.multi_level_conv_cls)):
            # print(i)
            added_branch_weight = self.bbox_head.multi_level_conv_cls[i].weight[self.ori_num_classes:, ...]
            added_branch_bias = self.bbox_head.multi_level_conv_cls[i].bias[self.ori_num_classes:, ...]
            state_dict['bbox_head.multi_level_conv_cls.{}.weight'.format(i)] = torch.cat(
                (state_dict['bbox_head.multi_level_conv_cls.{}.weight'.format(i)], added_branch_weight), dim=0)
            state_dict['bbox_head.multi_level_conv_cls.{}.bias'.format(i)] = torch.cat(
                (state_dict['bbox_head.multi_level_conv_cls.{}.bias'.format(i)], added_branch_bias), dim=0)
        # load state_dict
        if hasattr(self, 'module'):
            load_state_dict(self.module, state_dict, strict, logger)
        else:
            load_state_dict(self, state_dict, strict, logger)