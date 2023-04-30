from mmdet.models.builder import HEADS

from ..utils.usconv import set_channels, make_divisible
from .yolox_head_searchable import SearchableYOLOXHead
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      bias_init_with_prob)
from mmcv.ops.nms import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import (MlvlPointGenerator, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)

@HEADS.register_module()
class SearchableYOLOXHeadIncreResponse(SearchableYOLOXHead):
    def __init__(self, ori_num_classes, dist_loss_weight=1.0,**kwargs):

        super().__init__(**kwargs)
        self.ori_num_classes = ori_num_classes
        self.dist_loss_weight = dist_loss_weight
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def loss_new_class(self,
             cls_scores,
             bbox_preds,
             objectnesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """
        num_imgs = len(img_metas)
        ## add
        # print(gt_bboxes, gt_labels)
        # print(cls_scores[0].size(), bbox_preds[0].size(), objectnesses[0].size())
        cls_scores = [cls_score[:, self.ori_num_classes:, ...] for cls_score in cls_scores]
        
        # print(gt_labels[0])
        gt_labels = [gt_label-self.ori_num_classes for gt_label in gt_labels]
        
        ## end
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels-self.ori_num_classes)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        (pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets,
         num_fg_imgs) = multi_apply(
             self._get_target_single_new_class, flatten_cls_preds.detach(),
             flatten_objectness.detach(),
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_bboxes.detach(), gt_bboxes, gt_labels)

        # The experimental results show that ‘reduce_mean’ can improve
        # performance on the COCO dataset.
        num_pos = torch.tensor(
            sum(num_fg_imgs),
            dtype=torch.float,
            device=flatten_cls_preds.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        loss_bbox = self.loss_bbox(
            flatten_bboxes.view(-1, 4)[pos_masks],
            bbox_targets) / num_total_samples
        # print(flatten_objectness.view(-1, 1).size(), obj_targets.size(), )
        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1),
                                 obj_targets) / num_total_samples
        loss_cls = self.loss_cls(
            flatten_cls_preds.view(-1, self.num_classes-self.ori_num_classes)[pos_masks],
            cls_targets) / num_total_samples

        loss_dict = dict(
            loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj)

        if self.use_l1:
            loss_l1 = self.loss_l1(
                flatten_bbox_preds.view(-1, 4)[pos_masks],
                l1_targets) / num_total_samples
            loss_dict.update(loss_l1=loss_l1)

        return loss_dict
    
    @torch.no_grad()
    def _get_target_single_new_class(self, cls_preds, objectness, priors, decoded_bboxes,
                           gt_bboxes, gt_labels):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """

        num_priors = priors.size(0)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes-self.ori_num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return (foreground_mask, cls_target, obj_target, bbox_target,
                    l1_target, 0)

        # YOLOX uses center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes.
        offset_priors = torch.cat(
            [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

        assign_result = self.assigner.assign(
            cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid(),
            offset_priors, decoded_bboxes, gt_bboxes, gt_labels)

        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)

        pos_ious = assign_result.max_overlaps[pos_inds]
        # IOU aware classification score
        cls_target = F.one_hot(sampling_result.pos_gt_labels,
                               self.num_classes-self.ori_num_classes) * pos_ious.unsqueeze(-1)
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        l1_target = cls_preds.new_zeros((num_pos_per_img, 4))
        if self.use_l1:
            l1_target = self._get_l1_target(l1_target, bbox_target,
                                            priors[pos_inds])
        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1
        return (foreground_mask, cls_target, obj_target, bbox_target,
                l1_target, num_pos_per_img)
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def loss_old_class(self,
             cls_scores,
             bbox_preds,
             objectnesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """
        num_imgs = len(img_metas)
        ## add
        # print(gt_bboxes, gt_labels)
        # print(cls_scores[0].size(), bbox_preds[0].size(), objectnesses[0].size())
        cls_scores = [cls_score[:, :self.ori_num_classes, ...] for cls_score in cls_scores]
        
        # print(gt_labels[0])
        gt_labels = [gt_label for gt_label in gt_labels]
        
        ## end
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.ori_num_classes)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        (pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets,
         num_fg_imgs) = multi_apply(
             self._get_target_single_old_class, flatten_cls_preds.detach(),
             flatten_objectness.detach(),
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_bboxes.detach(), gt_bboxes, gt_labels)

        # The experimental results show that ‘reduce_mean’ can improve
        # performance on the COCO dataset.
        num_pos = torch.tensor(
            sum(num_fg_imgs),
            dtype=torch.float,
            device=flatten_cls_preds.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        loss_bbox = self.loss_bbox(
            flatten_bboxes.view(-1, 4)[pos_masks],
            bbox_targets) / num_total_samples
        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1),
                                 obj_targets) / num_total_samples
        loss_cls = self.loss_cls(
            flatten_cls_preds.view(-1, self.ori_num_classes)[pos_masks],
            cls_targets) / num_total_samples

        loss_dict = dict(
            loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj)

        if self.use_l1:
            loss_l1 = self.loss_l1(
                flatten_bbox_preds.view(-1, 4)[pos_masks],
                l1_targets) / num_total_samples
            loss_dict.update(loss_l1=loss_l1)

        return loss_dict
    
    @torch.no_grad()
    def _get_target_single_old_class(self, cls_preds, objectness, priors, decoded_bboxes,
                           gt_bboxes, gt_labels):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """

        num_priors = priors.size(0)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.ori_num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return (foreground_mask, cls_target, obj_target, bbox_target,
                    l1_target, 0)

        # YOLOX uses center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes.
        offset_priors = torch.cat(
            [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)
        # print(decoded_bboxes.size(), gt_bboxes.size())
        assign_result = self.assigner.assign(
            cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid(),
            offset_priors, decoded_bboxes, gt_bboxes, gt_labels)

        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)

        pos_ious = assign_result.max_overlaps[pos_inds]
        # IOU aware classification score
        cls_target = F.one_hot(sampling_result.pos_gt_labels,
                               self.ori_num_classes) * pos_ious.unsqueeze(-1)
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        l1_target = cls_preds.new_zeros((num_pos_per_img, 4))
        if self.use_l1:
            l1_target = self._get_l1_target(l1_target, bbox_target,
                                            priors[pos_inds])
        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1
        return (foreground_mask, cls_target, obj_target, bbox_target,
                l1_target, num_pos_per_img)
    
    def sel_pos(self, cls_scores, bbox_preds=None):
        """Select positive predictions based on classification scores.

        Args:
            model (nn.Module): The loaded detector.
            cls_scores (List[Tensor]): Classification scores for each FPN level.
            bbox_preds (List[Tensor]): BBox predictions for each FPN level.
            #centernesses (List[Tensor]): Centernesses predictions for each FPN level.

        Returns:
            cat_cls_scores (Tensor): FPN concatenated classification scores.
            #cat_centernesses (Tensor): FPN concatenated centernesses.
            topk_bbox_preds (Tensor): Selected top-k bbox predictions.
            topk_inds (Tensor): Selected top-k indices.
        """
        #assert len(cls_scores) == len(bbox_preds)
        if bbox_preds is not None:
            assert len(cls_scores) == len(bbox_preds)

        num_imgs = cls_scores[0].size(0)
        cat_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.ori_num_classes)
            for cls_score in cls_scores
        ]

        cat_cls_scores = torch.cat(cat_cls_scores, dim=1)

        cat_conf = cat_cls_scores.sigmoid()

        max_scores, _ = cat_conf.max(dim=-1)

        # cls_thr_0 = max_scores[0].mean() + 2 * max_scores[0].std()
        # valid_mask_0 = max_scores[0] > cls_thr_0
        # cls_inds_conf_0 = valid_mask_0.nonzero(as_tuple=False).squeeze(1)
        # topk_cls_scores_0 = cat_cls_scores[0].gather(
        #     0, cls_inds_conf_0.unsqueeze(-1).expand(-1, cat_cls_scores[0].size(-1)))

        # cls_thr_1 = max_scores[1].mean() + 2 * max_scores[1].std()
        # valid_mask_1 = max_scores[1] > cls_thr_1
        # cls_inds_conf_1 = valid_mask_1.nonzero(as_tuple=False).squeeze(1)
        # topk_cls_scores_1 = cat_cls_scores[1].gather(
        #     0, cls_inds_conf_1.unsqueeze(-1).expand(-1, cat_cls_scores[1].size(-1)))
        
        # topk_cls_scores = torch.cat((topk_cls_scores_0,topk_cls_scores_1),0)

        # topk_inds_cls_0 = cls_inds_conf_0
        # topk_inds_cls_1 = cls_inds_conf_1
        topk_cls_scores = []
        topk_inds_cls = []
        for bs in range(num_imgs):
            cls_thr_0 = max_scores[bs].mean() + 2 * max_scores[bs].std()
            valid_mask_0 = max_scores[bs] > cls_thr_0
            cls_inds_conf_0 = valid_mask_0.nonzero(as_tuple=False).squeeze(1)
            topk_cls_scores_0 = cat_cls_scores[bs].gather(
                0, cls_inds_conf_0.unsqueeze(-1).expand(-1, cat_cls_scores[bs].size(-1)))

            topk_cls_scores.append(topk_cls_scores_0)
            topk_inds_cls.append(cls_inds_conf_0)
        
        # topk_cls_scores = torch.cat(topk_cls_scores, 0)

        return topk_cls_scores, topk_inds_cls

    
    def loss_response(self, cls_scores, 
                        bbox_preds, 
                        objectnesses, 
                        cls_score_teacher, 
                        bbox_pred_teacher, 
                        objectness_teacher):
        
        topk_cls_scores_bs, topk_inds_cls, = self.sel_pos(cls_score_teacher)

        topk_cls_scores = torch.cat(topk_cls_scores_bs, 0)

        num_imgs = cls_scores[0].size(0)
        new_cls_scores = [
            cls_score[:, :self.ori_num_classes, :, :].permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.ori_num_classes)
            for cls_score in cls_scores
        ]
        new_cls_scores = torch.cat(new_cls_scores, dim=1)

        new_topk_cls_scores = []
        for bs in range(num_imgs):
            new_cls_scores_0 = new_cls_scores[bs].gather(
                    0, topk_inds_cls[bs].unsqueeze(-1).expand(-1, new_cls_scores[bs].size(-1)))
            new_topk_cls_scores.append(new_cls_scores_0)

        new_topk_cls_scores = torch.cat(new_topk_cls_scores, 0)
        loss_dist_cls = self.dist_loss_weight * \
            self.l2_loss(new_topk_cls_scores, topk_cls_scores)

        ## objectness
        
        new_flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]
        new_flatten_objectness = torch.cat(new_flatten_objectness, dim=1)

        old_flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectness_teacher
        ]
        old_flatten_objectness = torch.cat(old_flatten_objectness, dim=1)
        new_topk_objectness = []
        old_topk_objectness = []

        for bs in range(num_imgs):
            # print(new_flatten_objectness[bs].size(), topk_inds_cls[bs].size())
            new_objectness = new_flatten_objectness[bs].gather(
                    0, topk_inds_cls[bs])
            new_topk_objectness.append(new_objectness)

            old_objectness = old_flatten_objectness[bs].gather(
                    0, topk_inds_cls[bs])
            old_topk_objectness.append(old_objectness)
        
        new_topk_objectness = torch.cat(new_topk_objectness, 0)
        old_topk_objectness = torch.cat(old_topk_objectness, 0)

        old_topk_objectness = old_topk_objectness.view(-1, 1)

        loss_dist_obj = self.dist_loss_weight * self.loss_obj(new_topk_objectness.view(-1, 1),
                                 old_topk_objectness) / old_topk_objectness.size(0)
        
        loss_dist_bbox = self.get_kd_bbox_loss(new_topk_cls_scores, bbox_preds, \
                            topk_cls_scores_bs, bbox_pred_teacher, topk_inds_cls)

        return dict(loss_kd_cls=loss_dist_cls, loss_kd_obj=loss_dist_obj, loss_kd_bbox=loss_dist_bbox)
    
    def get_kd_bbox_loss(self, cls_scores, bbox_preds, cls_score_teacher, bbox_pred_teacher, topk_inds_cls):
        num_imgs = bbox_preds[0].size(0)

        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        # print(flatten_bbox_preds.size())

        flatten_teacher_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_pred_teacher
        ]
        flatten_teacher_bbox_preds = torch.cat(flatten_teacher_bbox_preds, dim=1)

        new_topk_bbox = []
        old_topk_bbox = []
        flatten_priors = []

        featmap_sizes = [bbox_pred.shape[2:] for bbox_pred in bbox_pred_teacher]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_pred_teacher[0].dtype,
            device=bbox_pred_teacher[0].device,
            with_stride=True)
        # print(len(mlvl_priors))
        # print(mlvl_priors[0].size())
        mlvl_priors = torch.cat(mlvl_priors)
        # print(mlvl_priors.size())

        for bs in range(num_imgs):
            # print(new_flatten_objectness[bs].size(), topk_inds_cls[bs].size())
            new_bbox = flatten_bbox_preds[bs].gather(
                    0, topk_inds_cls[bs].unsqueeze(-1).expand(-1, flatten_bbox_preds[bs].size(-1)))
            new_topk_bbox.append(new_bbox)

            old_bbox = flatten_teacher_bbox_preds[bs].gather(
                    0, topk_inds_cls[bs].unsqueeze(-1).expand(-1, flatten_teacher_bbox_preds[bs].size(-1)))
            old_topk_bbox.append(old_bbox)

            priors = mlvl_priors.gather(
                    0, topk_inds_cls[bs].unsqueeze(-1).expand(-1, mlvl_priors.size(-1)))
            flatten_priors.append(priors)

        decoded_bbox = []
        for bs in range(num_imgs):
            decoded_bbox.append(self._bbox_decode(flatten_priors[bs], old_topk_bbox[bs]))
            # print(decoded_bbox[-1].size())
        keeps = []
        nms_cfg=dict(iou_threshold=0.005) #0.005
        for bs in range(num_imgs):
            scores_cls = cls_score_teacher[bs].sigmoid()
            scores, labels = scores_cls.max(dim=-1)
            # print(scores.size(), labels.size())
            # print(decoded_bbox[bs].size())
            _, keep = batched_nms(decoded_bbox[bs][:,:4], scores.view(-1), labels.view(-1), nms_cfg)
            keeps.append(keep)
        
        # print(keeps)
        new_bbox_fliter = []
        old_bbox_fliter = []
        for bs in range(num_imgs):
            new_bbox_fliter.append(new_topk_bbox[bs][keeps[bs]])
            old_bbox_fliter.append(old_topk_bbox[bs][keeps[bs]])
        
        new_topk_bbox = torch.cat(new_topk_bbox, 0)
        old_topk_bbox = torch.cat(old_topk_bbox, 0)

        loss = self.dist_loss_weight * self.loss_bbox(new_topk_bbox, old_topk_bbox) / new_topk_bbox.size(0)

        return loss
    
    @staticmethod
    def l2_loss(pred, target, reduction='mean'):
        r"""Function that takes the mean element-wise square value difference.
        """
        assert target.size() == pred.size()
        loss = (pred - target).pow(2).float()
        if reduction != 'none':
            loss = torch.mean(loss) if reduction == 'mean' else torch.sum(loss)
        return loss
