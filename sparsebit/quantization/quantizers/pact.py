import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from sparsebit.quantization.quantizers import Quantizer as BaseQuantizer
from sparsebit.quantization.quantizers import register_quantizer
from sparsebit.quantization.common import QuantTarget, Granularity
from .quant_tensor import STE


@register_quantizer
class Quantizer(BaseQuantizer):
    TYPE = "PACT"

    def __init__(self, config):
        super(Quantizer, self).__init__(config)
        assert (
            self.qdesc.target == QuantTarget.FEATURE
        ), "PACT only support feature quantization"
        assert (
            not self.granularity == Granularity.LAYERWISE
        ), "PACT only supports per-tensor now!"
        self.init_alpha_value = config.QUANTIZER.PACT.ALPHA_VALUE

    def calc_qparams(self):
        if self.fake_fused:
            return self.scale, self.zero_point
        scale, zero_point = self.observer.calc_qparams()
        self.scale = self._broadcast_qparams(scale)
        self.zero_point = self._broadcast_qparams(zero_point)
        self.alpha = nn.Parameter(torch.Tensor([self.init_alpha_value]).to(self.device))
        return self.scale, self.zero_point

    def _qparams_preprocess(self, x):
        lower = (
            -self.alpha
            if self.qdesc.qmin < 0
            else torch.Tensor([0]).to(self.alpha.device)
        )
        self.lower = lower
        scale, zero_point = self.calc_qparams_with_minmax(lower, self.alpha.detach())
        return scale, zero_point

    def _forward(self, x, scale, zero_point=None):
        x_clamp = torch.clamp(x, self.lower, self.alpha)
        x_dq = STE.apply(x_clamp, scale, zero_point, self.qdesc, self.backend)
        return x_dq
