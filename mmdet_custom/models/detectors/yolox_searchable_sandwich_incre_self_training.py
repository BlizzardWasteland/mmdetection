# Copyright (c) OpenMMLab. All rights reserved.
import random
import torch
from mmcv.runner import get_dist_info, load_state_dict

from mmdet.models.builder import DETECTORS
from .yolox_searchable_sandwich import SearchableYOLOX_KD

from collections import OrderedDict
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDistributedDataParallel
import os
import mmcv
from mmdet.models import build_detector
from mmdet.models.detectors.single_stage import SingleStageDetector
# from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

@DETECTORS.register_module()
class SearchableYOLOX_KD_Incre_ST(SearchableYOLOX_KD):
    def __init__(self,
                 ori_num_classes,
                 ori_checkpoint_file,
                 ori_config_file,
                 anno_threshold,
                 iou_threshold,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.ori_num_classes = ori_num_classes
        # self.load_checkpoint_for_new_model(ori_checkpoint_file)
        # self.init_detector(ori_config_file, ori_checkpoint_file)
        self.ori_config_file = ori_config_file
        self.ori_checkpoint_file = ori_checkpoint_file
        self.anno_threshold = anno_threshold
        self.iou_threshold = iou_threshold
        # self.dist_loss_weight = dist_loss_weight
        # self.dist_bbox = dist_bbox
        # self.dist_cls = dist_cls
    
    def init_weights(self):
        super().init_weights()
        self.init_detector(self.ori_config_file, self.ori_checkpoint_file)

    def load_checkpoint_for_new_model(self, checkpoint_file, map_location='cpu', strict=False, logger=None):
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
        # print(state_dict['bbox_head.multi_level_conv_cls.0.bias'])
        # print(self.bbox_head.multi_level_conv_cls[0].bias.shape, self.bbox_head.multi_level_conv_cls[0].bias, state_dict['bbox_head.multi_level_conv_cls.0.bias'])
        
    
    def init_detector(self, config, checkpoint_file):
        """Initialize detector from config file.

        Args:
            config (str): Config file path or the config
                object.
            checkpoint_file (str): Checkpoint path. If left as None, the model
                will not load any weights.

        Returns:
            nn.Module: The constructed detector.
        """
        assert os.path.isfile(checkpoint_file), '{} is not a valid file'.format(checkpoint_file)
        ##### init original model & frozen it #####
        # build model
        cfg = mmcv.Config.fromfile(config)
        cfg.model.pretrained = None
        cfg.model.bbox_head.num_classes = self.ori_num_classes
        ori_model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))#test_cfg=cfg.test_cfg
        # load checkpoint
        load_checkpoint(ori_model, checkpoint_file)
        # set to eval mode
        ori_model.eval()
        # ori_model.forward = ori_model.forward_dummy
        # set requires_grad of all parameters to False
        for param in ori_model.parameters():
            param.requires_grad = False

        ##### init original branchs of new model #####
        self.load_checkpoint_for_new_model(checkpoint_file)

        self.ori_model = ori_model
    
    def teacher_model(self, img):
        """Inference image(s) with the detector.

        Args:
            model (nn.Module): The loaded detector.
            img (Tensor): Input to the model.

        Returns:
            outs (Tuple(List[Tensor])): Three model outputs.
                # cls_scores (List[Tensor]): Classification scores for each FPN level.
                # bbox_preds (List[Tensor]): BBox predictions for each FPN level.
                # centernesses (List[Tensor]): Centernesses predictions for each FPN level.
        """
        # forward the model without gradients
        with torch.no_grad():
            x = self.ori_model.extract_feat(img)
            outs = self.ori_model.bbox_head(x)

        return outs
    
    def forward_train(self, img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        # losses = dict()
        # print(self.bbox_head.multi_level_conv_cls[0].bias.shape, self.bbox_head.multi_level_conv_cls[0].bias)
        
        # new loss
        img, gt_bboxes = self._preprocess(img, gt_bboxes)

        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        outs = self.bbox_head(x)


        old_outs = self.teacher_model(img)
        results_teacher = self.ori_model.bbox_head.get_bboxes(
            *old_outs, img_metas=img_metas)
        
        # bboxes_teacher = [r[0][:, :4] for r in results_teacher]
        bboxes_teacher = [r[0] for r in results_teacher]
        labels_teacher = [r[1] for r in results_teacher]
        with torch.no_grad():
            for bs in range(len(gt_bboxes)):
                p_box = bboxes_teacher[bs]
                p_label = labels_teacher[bs]
                keep_1 = p_box[:, -1]>self.anno_threshold
                p_box = p_box[keep_1]
                p_label = p_label[keep_1]

                p_box = p_box[:, :4]
                gt_box = gt_bboxes[bs]
                if gt_box.shape[0]>0:
                    iou = bbox_overlaps(p_box, gt_box, use_legacy_coordinate=True)
                    # print(iou)
                    iou = iou.max(dim=1)[0]
                    # print(iou)
                    keep_2 = iou < self.iou_threshold

                    p_box = p_box[keep_2].detach()
                    p_label = p_label[keep_2].detach()

                    bboxes_teacher[bs] = p_box
                    labels_teacher[bs] = p_label

                # print(gt_labels[bs], gt_bboxes[bs])
                gt_bboxes[bs] = torch.cat([gt_bboxes[bs], p_box], dim=0)
                gt_labels[bs] = torch.cat([gt_labels[bs], p_label], dim=0)

                # print(gt_labels[bs], gt_bboxes[bs])

        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        # random resizing
        if (self._progress_in_iter + 1) % self._random_size_interval == 0:
            self._input_size = self._random_resize(device=img.device)
        self._progress_in_iter += 1

        return losses
    
    # def forward_train(self, img,
    #                   img_metas,
    #                   gt_bboxes,
    #                   gt_labels,
    #                   gt_bboxes_ignore=None):

    #     losses = dict()
    #     # print(self.bbox_head.multi_level_conv_cls[0].bias.shape, self.bbox_head.multi_level_conv_cls[0].bias)
        
    #     # new loss
    #     img, gt_bboxes = self._preprocess(img, gt_bboxes)

    #     super(SingleStageDetector, self).forward_train(img, img_metas)
    #     x = self.extract_feat(img)

    #     outs = self.bbox_head(x)
    #     loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
    #     losses_new = self.bbox_head.loss_new_class(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        
    #     losses.update(losses_new)

    #     cls_score, bbox_pred, objectness = outs
    #     # distillation loss

    #     old_outs = self.teacher_model(img)
    #     # cls_score_teacher, bbox_pred_teacher, objectness_teacher = old_outs
    #     results_teacher = self.ori_model.bbox_head.get_bboxes(
    #         *old_outs, img_metas=img_metas)
        
    #     # bboxes_teacher = [r[0][:, :4] for r in results_teacher]
    #     bboxes_teacher = [r[0] for r in results_teacher]
    #     # print(bboxes_teacher[0].size(), bboxes_teacher[0])
    #     labels_teacher = [r[1] for r in results_teacher]

    #     for bs in range(len(gt_bboxes)):
    #         p_box = bboxes_teacher[bs]
    #         p_label = labels_teacher[bs]
    #         keep_1 = p_box[:, -1]>self.anno_threshold
    #         p_box = p_box[keep_1]
    #         p_label = p_label[keep_1]

    #         p_box = p_box[:, :4]
    #         gt_box = gt_bboxes[bs]
    #         iou = bbox_overlaps(p_box, gt_box, use_legacy_coordinate=True)
    #         print(iou)
    #         iou = iou.max(dim=1)[0]
    #         print(iou)
    #         keep_2 = iou < self.iou_threshold

    #         p_box = p_box[keep_2]
    #         p_label = p_label[keep_2]

    #         bboxes_teacher[bs] = p_box
    #         labels_teacher[bs] = p_label

    #     assert len(bboxes_teacher)==len(gt_bboxes)
    #     loss_inputs_dist = outs + (bboxes_teacher, labels_teacher, img_metas)
    #     losses_old = self.bbox_head.loss_old_class(*loss_inputs_dist, gt_bboxes_ignore=None)

    #     losses['loss_dis_bbox'] = losses_old['loss_bbox'] * self.dist_loss_weight
    #     losses['loss_dis_obj'] = losses_old['loss_obj'] * self.dist_loss_weight
    #     if not self.dist_cls:
    #         losses['loss_dis_cls'] = losses_old['loss_cls'] * self.dist_loss_weight


    #     # random resizing
    #     if (self._progress_in_iter + 1) % self._random_size_interval == 0:
    #         self._input_size = self._random_resize(device=img.device)
    #     self._progress_in_iter += 1

    #     return losses
        
def bbox_overlaps(bboxes1,
                  bboxes2,
                  mode='iou',
                  eps=1e-6,
                  use_legacy_coordinate=False):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1 (ndarray): Shape (n, 4)
        bboxes2 (ndarray): Shape (k, 4)
        mode (str): IOU (intersection over union) or IOF (intersection
            over foreground)
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Note when function is used in `VOCDataset`, it should be
            True to align with the official implementation
            `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar`
            Default: False.

    Returns:
        ious (ndarray): Shape (n, k)
    """

    assert mode in ['iou', 'iof']
    if not use_legacy_coordinate:
        extra_length = 0.
    else:
        extra_length = 1.
    # bboxes1 = bboxes1.astype(np.float32)
    # bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols)).cuda(device=bboxes1.device)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = torch.zeros((cols, rows)).cuda(device=bboxes1.device)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + extra_length) * (
        bboxes1[:, 3] - bboxes1[:, 1] + extra_length)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + extra_length) * (
        bboxes2[:, 3] - bboxes2[:, 1] + extra_length)
    for i in range(bboxes1.shape[0]):
        x_start = torch.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = torch.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = torch.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = torch.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = torch.maximum(x_end - x_start + extra_length, torch.zeros((1)).cuda(bboxes1.device)) * torch.maximum(
            y_end - y_start + extra_length, torch.zeros((1)).cuda(bboxes1.device))
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = torch.maximum(union, eps * torch.ones((1)).cuda(bboxes1.device))
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.transpose(1,0)
    return ious