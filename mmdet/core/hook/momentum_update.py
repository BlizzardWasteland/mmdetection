# Copyright (c) OpenMMLab. All rights reserved.
import math
from math import cos, pi
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook

@HOOKS.register_module()
class MyMomentumUpdateHook(Hook):
    def __init__(self,
                 end_momentum=1.,
                 interval=1,
                 momentum_fun=None):
        # assert 0 < end_momentum < 1
        self.end_momentum = end_momentum
        self.interval = interval
        self.momentum_fun = momentum_fun

    def before_train_iter(self, runner):
        """To resume model with it's ema parameters more friendly.

        Register ema parameter as ``named_buffer`` to model.
        """
        assert hasattr(runner.model.module, 'momentum'), \
            "The runner must have attribute \"momentum\" in algorithms."
        assert hasattr(runner.model.module, 'base_momentum'), \
            "The runner must have attribute \"base_momentum\" in algorithms."

        if (runner.iter + 1) % self.interval != 0:
            return
        cur_iter = runner.iter
        max_iter = runner.max_iters
        base_m = runner.model.module.base_momentum
        m = self.end_momentum - (self.end_momentum - base_m) * (
            cos(pi * cur_iter / float(max_iter)) + 1) / 2
        runner.model.module.momentum = m

    def after_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        if (runner.iter + 1) % self.interval != 0:
            return
        # momentum = self.get_momentum(runner)
        if is_module_wrapper(runner.model):
            runner.model.module.momentum_update()
        else:
            runner.model.momentum_update()



# @HOOKS.register_module()
# class ExpMomentumEMAHook(BaseEMAHook):
#     """EMAHook using exponential momentum strategy.

#     Args:
#         total_iter (int): The total number of iterations of EMA momentum.
#            Defaults to 2000.
#     """

#     def __init__(self, total_iter=2000, **kwargs):
#         super(ExpMomentumEMAHook, self).__init__(**kwargs)
#         self.momentum_fun = lambda x: (1 - self.momentum) * math.exp(-(
#             1 + x) / total_iter) + self.momentum


# @HOOKS.register_module()
# class LinearMomentumEMAHook(BaseEMAHook):
#     """EMAHook using linear momentum strategy.

#     Args:
#         warm_up (int): During first warm_up steps, we may use smaller decay
#             to update ema parameters more slowly. Defaults to 100.
#     """

#     def __init__(self, warm_up=100, **kwargs):
#         super(LinearMomentumEMAHook, self).__init__(**kwargs)
#         self.momentum_fun = lambda x: min(self.momentum**self.interval,
#                                           (1 + x) / (warm_up + x))
