from mmdet.models.detectors import YOLOV3, BaseDetector
from mmdet.models.dense_heads.yolo_head import YOLOV3Head, constant_init, bias_init_with_prob, is_norm, normal_init, multiclass_nms
from mmdet.models.dense_heads.centernet_head import CenterNetHead
from mmdet.models.utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                     transpose_and_gather_feat)
from mmdet.core import (build_assigner, build_bbox_coder,
                        build_prior_generator, build_sampler, images_to_levels,
                        multi_apply, multiclass_nms)
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from mmdet.models import HEADS, build_loss, DETECTORS, SingleStageDetector
from mmcv.runner import force_fp32
from mmdet.core import bbox2result
import mmcv
import sys 
# sys.path.append("..") 

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from mmdet_custom.visualization import imshow_det_bboxes

@HEADS.register_module()
class YOLOV3PHead(YOLOV3Head):
    def __init__(self, feat_channel=None,
                 loss_center_heatmap=dict(
                     type='GaussianFocalLoss', loss_weight=1.0),
                 loss_offset=dict(type='L1Loss', loss_weight=1.0),
                 num_points=None,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.num_points = num_points

        in_channel = self.in_channels[-1]
        if feat_channel is None:
            feat_channel = in_channel

        self.heatmap_head = self._build_head(in_channel, feat_channel,
                                             num_points)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)

        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_offset = build_loss(loss_offset)
        
    def _build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def forward_point_single(self, feat):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the
               channels number is num_classes.
            offset_pred (Tensor): offset predicts, the channels number is 2.
        """
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        offset_pred = self.offset_head(feat)
        return center_heatmap_pred, offset_pred

    def forward_point(self, feats):
        return multi_apply(self.forward_point_single, feats[-1:])

    def forward(self, feats):
        list_pred_maps, = super().forward(feats)
        center_heatmap_pred, offset_pred = self.forward_point(feats)
        return list_pred_maps, center_heatmap_pred, offset_pred

    def init_weights(self):
        super().init_weights()
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        for m in self.offset_head.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            
    @force_fp32(apply_to=('center_heatmap_preds', 'offset_preds'))
    def loss_point(self,
             center_heatmap_preds,
             offset_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
               all levels with shape (B, num_classes, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
               with shape (B, 2, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_offset (Tensor): loss of offset heatmap.
        """
        assert len(center_heatmap_preds) == len(
            offset_preds) == 1
        center_heatmap_pred = center_heatmap_preds[0]
        offset_pred = offset_preds[0]

        target_result, avg_factor = self.get_point_targets(gt_bboxes, gt_labels,
                                                     center_heatmap_pred.shape,
                                                     img_metas[0]['batch_input_shape'])
        center_heatmap_target = target_result['center_heatmap_target']
        offset_target = target_result['offset_target']
        offset_target_weight = target_result['offset_target_weight']

        # Since the channel of offset_target is 2, the avg_factor
        # of loss_center_heatmap is always 1/2 of loss_offset.
        loss_center_heatmap = self.loss_center_heatmap(
            center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor)
        loss_offset = self.loss_offset(
            offset_pred,
            offset_target,
            offset_target_weight,
            avg_factor=avg_factor * 2)
        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_offset=loss_offset)

    def get_point_targets(self, gt_bboxes, gt_labels, feat_shape, img_shape):
        """Compute regression and classification targets in multiple images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (list[int]): feature map shape with value [B, _, H, W]
            img_shape (list[int]): image shape in [h, w] format.

        Returns:
            tuple[dict,float]: The float value is mean avg_factor, the dict has
               components below:
               - center_heatmap_target (Tensor): targets of center heatmap, \
                   shape (B, num_classes, H, W).
               - offset_target (Tensor): targets of offset predict, shape \
                   (B, 2, H, W).
               - offset_target_weight (Tensor): weights of offset \
                   predict, shape (B, 2, H, W).
        """
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_points, feat_h, feat_w])
        offset_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        offset_target_weight = gt_bboxes[-1].new_zeros(
            [bs, 2, feat_h, feat_w])

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            # print(center_x, center_y, width_ratio, height_ratio, img_w, img_h, feat_w, feat_h)
            gt_centers = torch.cat((center_x, center_y), dim=1)

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [ctx_int, cty_int], radius)

                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int

                offset_target_weight[batch_id, :, cty_int, ctx_int] = 1

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            offset_target=offset_target,
            offset_target_weight=offset_target_weight)
        return target_result, avg_factor

    @force_fp32(apply_to=('center_heatmap_preds', 'offset_preds'))
    def get_points(self,
                   center_heatmap_preds,
                   offset_preds,
                   img_metas,
                   rescale=True,
                   with_nms=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): Center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            offset_preds (list[Tensor]): Offset predicts for all levels
                with shape (B, 2, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.
            with_nms (bool): If True, do nms before return boxes.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(center_heatmap_preds) == len(offset_preds) == 1
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(
                self._get_point_single(
                    center_heatmap_preds[0][img_id:img_id + 1, ...],
                    offset_preds[0][img_id:img_id + 1, ...],
                    img_metas[img_id],
                    rescale=rescale,
                    with_nms=with_nms))
        return result_list

    def _get_point_single(self,
                           center_heatmap_pred,
                           offset_pred,
                           img_meta,
                           rescale=False,
                           with_nms=True):
        """Transform outputs of a single image into bbox results.

        Args:
            center_heatmap_pred (Tensor): Center heatmap for current level with
                shape (1, num_classes, H, W).
            offset_pred (Tensor): Offset for current level with shape
                (1, corner_offset_channels, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor, Tensor]: The first item is an (n, 5) tensor, where
                5 represent (tl_x, tl_y, br_x, br_y, score) and the score
                between 0 and 1. The shape of the second tensor in the tuple
                is (n,), and each element represents the class label of the
                corresponding box.
        """
        batch_det_points, batch_labels = self.decode_heatmap(
            center_heatmap_pred,
            offset_pred,
            img_meta['batch_input_shape'],
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel)

        det_points = batch_det_points.view([-1, 3])
        det_labels = batch_labels.view(-1)

        # batch_border = det_points.new_tensor(img_meta['border'])[...,
        #                                                          [2, 0]]
        # det_points[..., :2] -= batch_border

        if rescale:
            det_points[..., :2] /= det_points.new_tensor(
                img_meta['scale_factor'][..., :2])

        if with_nms:
            raise NotImplementedError
        return det_points, det_labels

    def decode_heatmap(self,
                       center_heatmap_pred,
                       offset_pred,
                       img_shape,
                       k=100,
                       kernel=3):
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            img_shape (list[int]): image shape in [h, w] format.
            k (int): Get top k center points from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 3)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
        """
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        offset = transpose_and_gather_feat(offset_pred, batch_index)
        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]
        c_x = topk_xs * (inp_w / width)
        c_y = topk_ys * (inp_h / height)

        batch_points = torch.stack([c_x, c_y], dim=2)
        batch_points = torch.cat((batch_points, batch_scores[..., None]),
                                 dim=-1)
        return batch_points, batch_topk_labels

    def get_gt_bboxes_points(self, raw_gt_bboxes, raw_gt_labels):
        gt_bboxes, gt_labels, gt_points, gt_plabels = [], [], [], []
        for raw_gt_bbox, raw_gt_label in zip(raw_gt_bboxes, raw_gt_labels):
            inds = raw_gt_label < self.num_classes
            gt_bboxes.append(raw_gt_bbox[inds, :])
            gt_labels.append(raw_gt_label[inds])
            inds = raw_gt_label >= self.num_classes
            gt_points.append(raw_gt_bbox[inds, :])
            gt_plabels.append(raw_gt_label[inds]-self.num_classes)
        return gt_bboxes, gt_labels, gt_points, gt_plabels

    @force_fp32(apply_to=('pred_maps', 'center_heatmap_preds', 'offset_preds'))
    def loss(self,
             pred_maps,
             center_heatmap_preds,
             offset_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        gt_bboxes, gt_labels, gt_points, gt_plabels = self.get_gt_bboxes_points(gt_bboxes, gt_labels)
        # for bbox
        loss_bbox = super().loss(pred_maps=pred_maps, 
            gt_bboxes=gt_bboxes[inds], gt_labels=gt_labels[inds], img_metas=img_metas, gt_bboxes_ignore=gt_bboxes_ignore)
        # for keypoint
        loss_center = self.loss_point(center_heatmap_preds=center_heatmap_preds, offset_preds=offset_preds, 
            gt_bboxes=gt_points, gt_labels=gt_plabels, img_metas=img_metas, gt_bboxes_ignore=gt_bboxes_ignore)
        return {**loss_bbox, **loss_center}

    def simple_test_points(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        outs = self.forward(feats)
        results_list = self.get_bboxes(
            *outs, img_metas=img_metas, rescale=rescale)
        return results_list

    def simple_test(self, feats, img_metas, rescale=False, require_points=False):
        list_pred_maps, center_heatmap_pred, offset_pred = self.forward(feats)
        bboxes = self.get_bboxes(
            list_pred_maps, img_metas=img_metas, rescale=rescale)
        points = self.get_points(center_heatmap_pred, offset_pred, img_metas, rescale=rescale)
        if require_points:
            # points = self.get_points(feats, img_metas, rescale=rescale)
            return (bboxes, points)
        else:
            return bboxes


def point2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 3), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]

@DETECTORS.register_module()
class YOLOV3P(YOLOV3):
    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        bboxes, points = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale, require_points=True)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bboxes
        ]
        point_results = [
            point2result(det_bboxes, det_labels, self.bbox_head.num_points)
            for det_bboxes, det_labels in points
        ]
        return list(zip(bbox_results, point_results))

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result[0], tuple):
            bbox_result, keypoint_results = [r[0] for r in result], [r[1] for r in result]
            # if isinstance(segm_result, tuple):
            #     segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, keypoint_results = result, None
        if isinstance(bbox_result, list):
            bbox_result = bbox_result[0]

        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        keypoints = None
        if keypoint_results is not None:
            if isinstance(keypoint_results, list):
                keypoint_results = keypoint_results[0]
            keypoints = np.vstack(keypoint_results)
            point_labels = [
            np.full(point.shape[0], i, dtype=np.int32)
                for i, point in enumerate(keypoint_results)
            ]
            point_labels = np.concatenate(point_labels)
            point_labels = point_labels + self.bbox_head.num_classes
            # keypoints = np.vstack(keypoint_results)
        # print(keypoints)
        # draw segmentation masks
        segms = None
        # if keypoints is not None and len(labels) > 0:  # non empty
        #     segms = mmcv.concat_list(segm_result)
        #     if isinstance(segms[0], torch.Tensor):
        #         segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        #     else:
        #         segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            keypoints,
            point_labels,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img