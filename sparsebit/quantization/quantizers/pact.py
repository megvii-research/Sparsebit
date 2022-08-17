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
        self.alpha = None
        self.alpha_value = config.QUANTIZER.PACT.ALPHA_VALUE
        self.init_params = False

    def calc_qparams(self):
        if self.fake_fused:
            return self.scale, self.zero_point
        if not self.init_params:
            self.alpha = nn.Parameter(torch.Tensor([self.alpha_value]).to(self.device))
            self.init_params = True
        return self.scale, self.zero_point

    def _forward(self, x):
        if self.fake_fused:
            return x

        if self.qdesc.qmin < 0:
            x_clamp = torch.clamp(x, -self.alpha, self.alpha)
            min_val = -self.alpha.detach()
        else:
            x_clamp = torch.clamp(x, torch.Tensor([0]).to(self.device), self.alpha)
            min_val = torch.Tensor([0]).to(self.device)

        self.update_observer(self.alpha.detach())
        self.update_observer(min_val)
        self.scale, self.zero_point = self.observer.calc_qparams()

        x_dq = STE.apply(x_clamp, self.scale, self.zero_point, self.qdesc, self.backend)
        return x_dq
