from types import MethodType
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.models.utils import make_divisible
from .kd_loss import *

class SearchBase:
    def __init__(self, bn_training_mode=True, inplace=None, kd_weight=1e-8, divisor=8) -> None:
        self.inplace = inplace
        self.arch = None
        self.archs = None
        self.kd_weight = kd_weight
        # 不同蒸馏对应的loss计算方法
        if self.inplace == 'L2':
            self.kd_loss = DL2()
        elif self.inplace == 'L2Softmax':
            self.kd_loss = DL2(softmax=True)
        elif self.inplace == 'DML':
            self.kd_loss = DML()
        elif self.inplace == 'NonLocal':
            self.kd_loss = NonLocalBlockLoss(self.neck.out_channels, 64)
        self.divisor = divisor
        self.bn_training_mode = bn_training_mode

        for name, module in self.named_modules():
            self.add_nas_attrs(module)

    @staticmethod
    def modify_conv_forward(module):
        """Modify the forward method of a conv layer."""
        def modified_forward(self, feature):
            assert self.groups == 1
            weight = self.weight[:self.out_channels, :self.in_channels, :, :]
            if self.bias is not None:
                bias = self.bias[:self.out_channels]
            else:
                bias = self.bias
            return self._conv_forward(feature, weight, bias)

        return MethodType(modified_forward, module)

    @staticmethod
    def modify_fc_forward(module):
        """Modify the forward method of a linear layer."""
        def modified_forward(self, feature):
            weight = self.weight[:self.out_features, :self.in_features]
            if self.bias is not None:
                bias = self.bias[:self.out_features]
            else:
                bias = self.bias 
            return F.linear(feature, weight, bias)

        return MethodType(modified_forward, module)

    @staticmethod
    def modify_seq_forward(module):
        """Modify the forward method of a sequential container."""
        def modified_forward(self, input):
            for module in self[:self.num_layers]:
                input = module(input)
            return input

        return MethodType(modified_forward, module)
    
    @staticmethod
    def modify_bn_forward(module):
        """Modify the forward method of a linear layer."""
        def modified_forward(self, feature):
            self._check_input_dim(feature)
            # exponential_average_factor is set to self.momentum
            # (when it is available) only so that it gets updated
            # in ONNX graph when this node is exported to ONNX.
            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum

            if self.training and self.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if self.num_batches_tracked is not None:  # type: ignore[has-type]
                    self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum

            r"""
            Decide whether the mini-batch stats should be used for normalization rather than the buffers.
            Mini-batch stats are used in training mode, and in eval mode when buffers are None.
            """
            if self.training:
                bn_training = True
            else:
                bn_training = (self.running_mean is None) and (self.running_var is None)

            r"""
            Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
            passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
            used for normalization (i.e. in eval mode when buffers are not None).
            """
            return F.batch_norm(
                feature,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean[:self.num_features]
                if not self.training or self.track_running_stats
                else None,
                self.running_var[:self.num_features] if not self.training or self.track_running_stats else None,
                self.weight[:self.num_features],
                self.bias[:self.num_features],
                bn_training,
                exponential_average_factor,
                self.eps,
            )

        return MethodType(modified_forward, module)
    
    def add_nas_attrs(self, module):
        """Add masks to a ``nn.Module``."""
        if isinstance(module, nn.Conv2d):
            module.forward = self.modify_conv_forward(module)
        if isinstance(module, nn.Linear):
            module.forward = self.modify_fc_forward(module)
        if isinstance(module, _BatchNorm):
            module.forward = self.modify_bn_forward(module)
        if isinstance(module, nn.Sequential):
            module.num_layers = len(module)
            module.forward = self.modify_seq_forward(module)

    def set_archs(self, archs, **kwargs):
        self.archs = archs

    def set_arch(self, arch, **kwargs):
        self.arch = arch
        self.backbone.set_arch(self.arch)
        self.neck.set_arch(self.arch)
        self.bbox_head.set_arch(self.arch)

    def train(self, mode=True):
        """Overwrite the train method in `nn.Module` to set `nn.BatchNorm` to
        training mode when model is set to eval mode when
        `self.bn_training_mode` is `True`.
        Args:
            mode (bool): whether to set training mode (`True`) or evaluation
                mode (`False`). Default: `True`.
        """
        super().train(mode)
        if not mode and self.bn_training_mode:
            for module in self.modules():
                if isinstance(module, _BatchNorm):
                    module.training = True
