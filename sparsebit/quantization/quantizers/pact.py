import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from sparsebit.quantization.quantizers import Quantizer as BaseQuantizer
from sparsebit.quantization.quantizers import register_quantizer
from sparsebit.quantization.common import QuantTarget
from .quant_tensor import STE


@register_quantizer
class Quantizer(BaseQuantizer):
    TYPE = "PACT"

    def __init__(self, config):
        super(Quantizer, self).__init__(config)
        assert (
            self.qdesc.target == QuantTarget.FEATURE
        ), "PACT only support feature quantization"
        assert not self.qdesc.is_perchannel, "PACT no yet supports per-channel"
        if not self.fake_fused:
            self.alpha = nn.Parameter(
                torch.Tensor([config.QUANTIZER.PACT.ALPHA_VALUE]).to(self.device)
            )

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
