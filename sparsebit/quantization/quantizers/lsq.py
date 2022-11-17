import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import warnings

from sparsebit.quantization.quantizers import Quantizer as BaseQuantizer
from sparsebit.quantization.quantizers import register_quantizer
from .quant_tensor import STE


class gs_scaling(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ratio):
        ctx.ratio = ratio
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.ratio, None


@register_quantizer
class Quantizer(BaseQuantizer):
    TYPE = "LSQ"

    def __init__(self, config):
        super(Quantizer, self).__init__(config)
        self.init_params = False  # LSQ需要基于calibration做初始化

    def calc_qparams(self):
        if self.fake_fused:
            return self.scale, self.zero_point
        x_oc = self.observer.get_calibration_data(c_first=True)
        if x_oc.min()<0 and not self.qdesc.is_symmetric:
            warnings.warn("Found data less than 0, reset quantizer scheme as symmetric")
            self.qdesc.set_symmetric(True)
        if not self.init_params:
            if self.is_perchannel:
                scale = 2 * x_oc.abs().mean(axis=1) / math.sqrt(self.qdesc.qmax)
            else:
                scale = 2 * x_oc.abs().mean() / math.sqrt(self.qdesc.qmax)
            self.scale = nn.Parameter(self._broadcast_qparams(scale.to(self.device)))
            self.zero_point = self._broadcast_qparams(torch.zeros_like(self.scale))
            self.init_params = True
        return self.scale, self.zero_point

    def _qparams_preprocess(self, x):
        scale = self.scale.abs()
        zero_point = torch.clamp(self.zero_point, self.qdesc.qmin, self.qdesc.qmax)
        return scale, zero_point

    def _forward(self, x, scale, zero_point):
        if self.is_perchannel:
            num_perchannel = x.numel() / x.shape[self.qdesc.ch_axis]
            gs_ratio = 1.0 / math.sqrt(num_perchannel * self.qdesc.qmax)
        else:
            gs_ratio = 1.0 / math.sqrt(x.numel() * self.qdesc.qmax)
        scale = gs_scaling.apply(scale, gs_ratio)
        x_dq = STE.apply(x, scale, zero_point, self.qdesc, self.backend)
        return x_dq
