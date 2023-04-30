# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch

from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.core.bbox.coder.yolo_bbox_coder import YOLOBBoxCoder

@BBOX_CODERS.register_module()
class RRYOLOBBoxCoder(YOLOBBoxCoder):
    """YOLO BBox coder.
    Following `YOLO <https://arxiv.org/abs/1506.02640>`_, this coder divide
    image into grids, and encode bbox (x1, y1, x2, y2) into (cx, cy, dw, dh).
    cx, cy in [0., 1.], denotes relative center position w.r.t the center of
    bboxes. dw, dh are the same as :obj:`DeltaXYWHBBoxCoder`.
    Args:
        eps (float): Min value of cx, cy when encoding.
    """

    def __init__(self, eps=1e-6, scale_x_y=1.0):
        super(RRYOLOBBoxCoder, self).__init__(eps=eps)
        self.scale_x_y = scale_x_y

    def decode(self, bboxes, pred_bboxes, stride):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            boxes (torch.Tensor): Basic boxes, e.g. anchors.
            pred_bboxes (torch.Tensor): Encoded boxes with shape
            stride (torch.Tensor | int): Strides of bboxes.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        # print(pred_bboxes.shape, bboxes.shape)
        # assert pred_bboxes.size(0) == bboxes.size(0)
        assert pred_bboxes.size(-1) == bboxes.size(-1) == 4
        x_center = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
        y_center = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
        w = bboxes[..., 2] - bboxes[..., 0]
        h = bboxes[..., 3] - bboxes[..., 1]
        # Get outputs x, y
        x_center_pred = (pred_bboxes[..., 0] * self.scale_x_y - 0.5 * (self.scale_x_y - 1) - 0.5) * stride + x_center
        y_center_pred = (pred_bboxes[..., 1] * self.scale_x_y - 0.5 * (self.scale_x_y - 1) - 0.5) * stride + y_center
        w_pred = torch.exp(pred_bboxes[..., 2]) * w
        h_pred = torch.exp(pred_bboxes[..., 3]) * h

        decoded_bboxes = torch.stack(
            (x_center_pred - w_pred / 2, y_center_pred - h_pred / 2,
             x_center_pred + w_pred / 2, y_center_pred + h_pred / 2),
            dim=-1)

        return decoded_bboxes

    # @mmcv.jit(coderize=True)
    # def decode(self, bboxes, pred_bboxes, stride):
    #     """Apply transformation `pred_bboxes` to `boxes`.
    #     Args:
    #         boxes (torch.Tensor): Basic boxes, e.g. anchors.
    #         pred_bboxes (torch.Tensor): Encoded boxes with shape
    #         stride (torch.Tensor | int): Strides of bboxes.
    #     Returns:
    #         torch.Tensor: Decoded boxes.
    #     """
    #     assert pred_bboxes.size(-1) == bboxes.size(-1) == 4
    #     xy_centers = (bboxes[..., :2] + bboxes[..., 2:]) * 0.5 + (
    #         pred_bboxes[..., :2] - 0.5) * stride
    #     whs = (bboxes[..., 2:] -
    #            bboxes[..., :2]) * 0.5 * pred_bboxes[..., 2:].exp()
    #     decoded_bboxes = torch.stack(
    #         (xy_centers[..., 0] - whs[..., 0], xy_centers[..., 1] -
    #          whs[..., 1], xy_centers[..., 0] + whs[..., 0],
    #          xy_centers[..., 1] + whs[..., 1]),
    #         dim=-1)
    #     return decoded_bboxes