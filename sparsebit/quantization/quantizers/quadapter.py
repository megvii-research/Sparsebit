import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from sparsebit.quantization.quantizers import Quantizer as BaseQuantizer
from sparsebit.quantization.quantizers import register_quantizer
from .quant_tensor import STE


@register_quantizer
class Quantizer(BaseQuantizer):
    TYPE = "Quadapter"

    def __init__(self, config):
        super(Quantizer, self).__init__(config)
        self.alpha = None

    def init_alpha(self, x: torch.Tensor):
        alpha_shape = [1 for _ in range(self.dims)]
        alpha_shape[self.qdesc._ch_axis] = x.shape[self.qdesc._ch_axis]
        self.alpha = nn.Parameter(torch.ones(alpha_shape).to(self.device))

    def update_observer(self, x):
        self.dims = len(x.shape)
        self.observer.data_cache.update(x.detach())
        if self.alpha is None:
            self.init_alpha(x)

    def _forward(self, x_f, scale, zero_point):
        import ipdb
        ipdb.set_trace()
        x_f = x_f * self.alpha
        x_dq = STE.apply(x_f, scale, zero_point, self.qdesc, self.backend)
        x_dq = x_dq / self.alpha
        return x_dq
