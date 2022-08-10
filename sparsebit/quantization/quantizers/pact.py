import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from sparsebit.quantization.quantizers import Quantizer as BaseQuantizer
from sparsebit.quantization.quantizers import register_quantizer
from .quant_tensor import STE


class pact_ste(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, Qn, Qp):
        with_neg = Qn < 0
        min_val = -alpha.item() if with_neg else 0
        max_val = alpha.item()
        Qn_tensor = torch.tensor(Qn)
        ctx.save_for_backward(x, alpha, Qn_tensor)
        y = torch.clamp(x, min=min_val, max=max_val)
        return y

    @staticmethod
    def backward(ctx, dLdy):
        x, alpha, Qn = ctx.saved_tensors
        with_neg = Qn.item() < 0
        min_val = -alpha.item() if with_neg else 0
        max_val = alpha.item()
        lower_bound = x < min_val
        upper_bound = x > max_val
        x_mask = ~(lower_bound | upper_bound)
        grad_x = dLdy * x_mask.float()
        grad_alpha = torch.sum(dLdy * torch.ge(x, max_val).float()).view(-1)
        if with_neg:
            grad_alpha += torch.sum(dLdy * torch.le(x, min_val).float()).view(-1)
        return grad_x, grad_alpha, None, None


@register_quantizer
class Quantizer(BaseQuantizer):
    TYPE = "PACT"

    def __init__(self, config):
        super(Quantizer, self).__init__(config)
        self.alpha = None
        self.init_params = False

    def calc_qparams(self):
        if self.fake_fused:
            return self.scale, self.zero_point
        if not self.init_params:
            self.alpha = nn.Parameter(torch.ones(1).to(self.device) * 10)
            self.init_params = True
        return self.scale, self.zero_point

    def _forward(self, x):
        if self.fake_fused:
            return x

        x_clamp = pact_ste.apply(x, self.alpha, self.qdesc.qmin, self.qdesc.qmax)
        self.scale = self.alpha.detach() / (
            (self.qdesc.qmax - self.qdesc.qmin) // 2
            if self.qdesc.qmin < 0
            else self.qdesc.qmax - self.qdesc.qmin
        )
        x_dq = STE.apply(x_clamp, self.scale, self.zero_point, self.qdesc, self.backend)
        return x_dq
