import torch

from mmdet.models.builder import HEADS
from mmdet.models.dense_heads import YOLOXHead

@HEADS.register_module()
class ScaledYOLOXHead(YOLOXHead):

    def _bbox_decode(self, priors, bbox_preds):
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        whs = (bbox_preds[..., 2:] / 10).exp() * priors[:, 2:] # to reduce quantization error

        tl_x = (xys[..., 0] - whs[..., 0] / 2)
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)

        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        return decoded_bboxes
