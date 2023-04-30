# Copyright (c) OpenMMLab. All rights reserved.
import random

from mmcv.runner import get_dist_info

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.yolox import YOLOX


@DETECTORS.register_module()
class YOLOX_KD(YOLOX):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 input_size=(640, 640),
                 size_multiplier=32,
                 random_size_range=(15, 25),
                 random_size_interval=10,
                 init_cfg=None,
                 is_kd=False,
                 quant_cfg=None):
        super(YOLOX_KD, self).__init__(backbone, neck, bbox_head, train_cfg,test_cfg, pretrained,\
            input_size,size_multiplier,random_size_range,random_size_interval,init_cfg=init_cfg)
        self.is_kd = is_kd
        self.quant_cfg = quant_cfg

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        # Multi-scale training
        img, gt_bboxes = self._preprocess(img, gt_bboxes)

        losses = super(YOLOX, self).forward_train(img, img_metas, gt_bboxes,
                                                  gt_labels, gt_bboxes_ignore)

        # random resizing
        if (self._progress_in_iter + 1) % self._random_size_interval == 0:
            self._input_size = self._random_resize(device=img.device)
        self._progress_in_iter += 1

        if self.is_kd:
            return losses, img, gt_bboxes
        else:
            return losses