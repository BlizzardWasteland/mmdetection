from mmdet.models.necks.yolox_pafpn import YOLOXPAFPN
from mmdet.models.builder import NECKS
import torch

@NECKS.register_module()
class YOLOXPAFPNv3(YOLOXPAFPN):
    def forward(self, inputs):
        outs = super().forward(inputs)
        return outs[::-1]