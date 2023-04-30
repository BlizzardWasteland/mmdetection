import torch

from mmdet.models.builder import DETECTORS, build_head
from mmdet.models.detectors.single_stage import BaseDetector
from mmdet.models.detectors import YOLOV3, YOLOX

from mmdet.core import bbox2result
import numpy as np

def get_gt_bboxes_points(raw_gt_bboxes, raw_gt_labels, num_classes):
    gt_bboxes, gt_labels, gt_points, gt_plabels = [], [], [], []
    for raw_gt_bbox, raw_gt_label in zip(raw_gt_bboxes, raw_gt_labels):
        inds = raw_gt_label < num_classes
        gt_bboxes.append(raw_gt_bbox[inds, :])
        gt_labels.append(raw_gt_label[inds])
        inds = raw_gt_label >= num_classes
        gt_points.append(raw_gt_bbox[inds, :])
        gt_plabels.append(raw_gt_label[inds]-num_classes)
    return gt_bboxes, gt_labels, gt_points, gt_plabels

def point2result(points, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        points (torch.Tensor | np.ndarray): shape (n, 3)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if points.shape[0] == 0:
        return [np.zeros((0, 3), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        return [points[labels == i, :] for i in range(num_classes)]

class PointBase:
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

        bboxes = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bboxes
        ]

        points = self.point_head.simple_test(
            feat, img_metas, rescale=rescale)
        point_results = [
            point2result(det_points, det_labels, self.point_head.num_classes)
            for det_points, det_labels in points
        ]
        
        return list(zip(bbox_results, point_results))

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
        BaseDetector.forward_train(self, img, img_metas)
        x = self.extract_feat(img)
        gt_bboxes, gt_labels, gt_points, gt_plabels = get_gt_bboxes_points(gt_bboxes, gt_labels, self.bbox_head.num_classes)
        losses_bbox = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        losses_point = self.point_head.forward_train(x, img_metas, gt_points, gt_plabels, None)
        return {**losses_bbox, **losses_point}

@DETECTORS.register_module()
class Point_YOLOV3(PointBase, YOLOV3):
    def __init__(self, point_head, **kwargs) -> None:
        YOLOV3.__init__(self, **kwargs)
        self.point_head = build_head(point_head)
        

@DETECTORS.register_module()
class Point_YOLOX(PointBase, YOLOX):
    def __init__(self, point_head, **kwargs) -> None:
        YOLOX.__init__(self, **kwargs)
        self.point_head = build_head(point_head)
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        # Multi-scale training
        img, gt_bboxes = self._preprocess(img, gt_bboxes)

        losses = super(Point_YOLOX, self).forward_train(img, img_metas, gt_bboxes,
                                                  gt_labels, gt_bboxes_ignore)

        # random resizing
        if (self._progress_in_iter + 1) % self._random_size_interval == 0:
            self._input_size = self._random_resize()
        self._progress_in_iter += 1

        return losses