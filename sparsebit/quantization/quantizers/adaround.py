import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from sparsebit.quantization.quantizers import Quantizer as BaseQuantizer
from sparsebit.quantization.quantizers import register_quantizer
from .quant_tensor import STE


@register_quantizer
class Quantizer(BaseQuantizer):
    TYPE = "Adaround"

    def __init__(self, config):
        super(Quantizer, self).__init__(config)
        self.alpha = None
        self.soft_targets = False
        self.round_mode = "learned_hard_sigmoid"

    def calc_qparams(self):
        if self.fake_fused:
            return self.scale, self.zero_point
        scale, zero_point = self.observer.calc_qparams()
        self.scale = self._broadcast_qparams(scale)
        self.zero_point = self._broadcast_qparams(zero_point)
        self.gamma, self.zeta = -0.1, 1.1
        self.init_alpha(x=self.weight.data.clone())
        return self.scale, self.zero_point

    def init_alpha(self, x: torch.Tensor):
        x_floor = torch.floor(x / self.scale)
        if self.round_mode == "learned_hard_sigmoid":
            rest = (x / self.scale) - x_floor  # rest of rounding [0, 1)
            alpha = -torch.log(
                (self.zeta - self.gamma) / (rest - self.gamma) - 1
            )  # => sigmoid(alpha) = rest
            self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError

    def update_observer(self, weight):
        self.weight = weight.detach()
        self.dims = len(weight.shape)
        self.observer.data_cache.update(self.weight)

    def get_soft_targets(self):
        return torch.clamp(
            torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1
        )

    def _forward(self, x):
        if self.round_mode == "learned_hard_sigmoid":
            assert self.alpha is not None
            x_floor = torch.floor(x / self.scale)
            if self.soft_targets:
                x_int = x_floor + self.get_soft_targets()
            else:
                x_int = x_floor + (self.alpha >= 0).float()

            x_q = x_int + self.zero_point
            x_q = torch.clamp(x_q, self.qdesc.qmin, self.qdesc.qmax)

        x_dq = (x_q - self.zero_point) * self.scale
        return x_dq