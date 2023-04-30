from mmdet.models.builder import HEADS

from ..utils.usconv import set_channels, make_divisible
# from .yolox_head_scaled import ScaledYOLOXHead
# from mmdet.models.dense_heads.yolox_head import YOLOXHead
from mmdet.models.dense_heads.yolox_cos_head import YOLOXCosHead


# @HEADS.register_module()
# class SearchableYOLOXHead(ScaledYOLOXHead):
#     def set_arch(self, arch, divisor=8):
#         widen_factor_out_neck = arch['widen_factor_neck_out']
#         if isinstance(widen_factor_out_neck, (int, float)):
#             widen_factor_out_neck = [widen_factor_out_neck]*2
#         in_channel = make_divisible(self.in_channels * widen_factor_out_neck[0], divisor)
#         feat_channels = make_divisible(self.in_channels * widen_factor_out_neck[1], divisor)
        
#         set_channels(self.multi_level_cls_convs, feat_channels, feat_channels)
#         set_channels(self.multi_level_reg_convs, feat_channels, feat_channels)

#         set_channels(self.multi_level_conv_cls, in_channels=feat_channels)
#         set_channels(self.multi_level_conv_reg, in_channels=feat_channels)
#         set_channels(self.multi_level_conv_obj, in_channels=feat_channels)
#         for i, _ in enumerate(self.strides):
#             self.multi_level_cls_convs[i][0].conv.in_channels = in_channel
#             self.multi_level_reg_convs[i][0].conv.in_channels = in_channel

@HEADS.register_module()
class SearchableYOLOXHeadCos(YOLOXCosHead):
    def set_arch(self, arch, divisor=8):
        widen_factor_out_neck = arch['widen_factor_neck_out']
        if isinstance(widen_factor_out_neck, (int, float)):
            widen_factor_out_neck = [widen_factor_out_neck]*2
        in_channel = make_divisible(self.in_channels * widen_factor_out_neck[0], divisor)
        feat_channels = make_divisible(self.in_channels * widen_factor_out_neck[1], divisor)
        
        set_channels(self.multi_level_cls_convs, feat_channels, feat_channels)
        set_channels(self.multi_level_reg_convs, feat_channels, feat_channels)

        set_channels(self.multi_level_conv_cls, in_channels=feat_channels)
        set_channels(self.multi_level_conv_reg, in_channels=feat_channels)
        set_channels(self.multi_level_conv_obj, in_channels=feat_channels)
        for i, _ in enumerate(self.strides):
            self.multi_level_cls_convs[i][0].conv.in_channels = in_channel
            self.multi_level_reg_convs[i][0].conv.in_channels = in_channel
