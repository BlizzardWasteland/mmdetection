# Copyright (c) OpenMMLab. All rights reserved.
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.runner import get_dist_info

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.yolox import YOLOX
from ..algorithm.searchbase import SearchBase

# @DETECTORS.register_module()
class SearchableYOLOX(YOLOX, SearchBase):
    r"""Implementation of `YOLOX: Exceeding YOLO Series in 2021
    <https://arxiv.org/abs/2107.08430>`_

    Note: Considering the trade-off between training speed and accuracy,
    multi-scale training is temporarily kept. More elegant implementation
    will be adopted in the future.

    Args:
        backbone (nn.Module): The backbone module.
        neck (nn.Module): The neck module.
        bbox_head (nn.Module): The bbox head module.
        train_cfg (obj:`ConfigDict`, optional): The training config
            of YOLOX. Default: None.
        test_cfg (obj:`ConfigDict`, optional): The testing config
            of YOLOX. Default: None.
        pretrained (str, optional): model pretrained path.
            Default: None.
        input_size (tuple): The model default input image size.
            Default: (640, 640).
        size_multiplier (int): Image size multiplication factor.
            Default: 32.
        random_size_range (tuple): The multi-scale random range during
            multi-scale training. The real training image size will
            be multiplied by size_multiplier. Default: (15, 25).
        random_size_interval (int): The iter interval of change
            image size. Default: 10.
        init_cfg (dict, optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 *args,
                 bn_training_mode,
                 inplace=False, # distill
                 kd_weight=1e-8,
                 divisor=8,
                 search=False,
                 **kwargs
                 ):
        YOLOX.__init__(self, *args, **kwargs)
        SearchBase.__init__(self, bn_training_mode=bn_training_mode, inplace=inplace, kd_weight=kd_weight, divisor=divisor)
        self.search = search

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # Multi-scale training
        img, gt_bboxes = self._preprocess(img, gt_bboxes)

        losses = dict()
        if not isinstance(self.archs, list): # not sandwich
            self.archs = [self.arch]

        for idx, arch in enumerate(self.archs):
            if arch is not None:
                self.set_arch(arch)

            x = self.extract_feat(img)

            if len(self.archs) > 1 and self.inplace: # inplace distill
                if idx == 0: # 最大的子网
                    teacher_feat = x
                else:
                    kd_feat_loss = 0
                    student_feat = x
                    for i in range(len(student_feat)):
                        kd_feat_loss += self.kd_loss(student_feat[i], teacher_feat[i].detach(), i) * self.kd_weight

                    losses.update({'kd_feat_loss_{}'.format(idx): kd_feat_loss})

            head_loss = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)

            losses.update({'loss_cls_{}'.format(idx): head_loss['loss_cls']})
            losses.update({'loss_bbox_{}'.format(idx): head_loss['loss_bbox']})
            losses.update({'loss_obj_{}'.format(idx): head_loss['loss_obj']})

        # random resizing
        if (self._progress_in_iter + 1) % self._random_size_interval == 0:
            self._input_size = self._random_resize()
        self._progress_in_iter += 1

        return losses
