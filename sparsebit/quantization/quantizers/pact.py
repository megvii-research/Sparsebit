import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from sparsebit.quantization.quantizers import Quantizer as BaseQuantizer
from sparsebit.quantization.quantizers import register_quantizer
from .quant_tensor import STE


@register_quantizer
class Quantizer(BaseQuantizer):
    TYPE = "PACT"

    def __init__(self, config):
        super(Quantizer, self).__init__(config)
        assert not self.qdesc.is_perchannel, "PACT no yet supports per-channel"
        if not self.fake_fused:
            self.alpha = nn.Parameter(
                torch.Tensor([config.QUANTIZER.PACT.ALPHA_VALUE]).to(self.device)
            )

    def _forward(self, x):
        if self.fake_fused:
            return x
        lower = (
            -self.alpha if self.qdesc.qmin < 0 else torch.Tensor([0]).to(self.device)
        )
        x_clamp = torch.clamp(x, lower, self.alpha)
        self.scale, self.zero_point = self.calc_qparams_with_minmax(
            lower, self.alpha.detach()
        )
        x_dq = STE.apply(x_clamp, self.scale, self.zero_point, self.qdesc, self.backend)
        return x_dq
