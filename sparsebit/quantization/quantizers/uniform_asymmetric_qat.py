import torch
import torch.nn as nn
from sparsebit.quantization.quantizers import Quantizer as BaseQuantizer
from sparsebit.quantization.quantizers import register_quantizer
from .quant_tensor import STE


class STE_Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, grad):
        return grad, None

@register_quantizer
class Quantizer(BaseQuantizer):
    TYPE = "uniform_asymmetric_qat"

    def __init__(self, config):
        super(Quantizer, self).__init__(config)
        self.init_params = False  # uniform_qat需要基于calibration做初始化
        self.running_stat = False

    def calc_qparams(self, bit_allocation):
        if self.fake_fused:
            return self.scale, self.zero_point
        scale, zero_point = self.observer.calc_qparams(bit_allocation)
        self.max_val = self.observer.max_val.to(self.device)
        self.min_val = self.observer.min_val.to(self.device)
        self.init_params = True
        return self.scale, self.zero_point

    def _qparams_preprocess(self, x):
        with torch.no_grad():
            if self.running_stat:
                x_max = x.max()
                x_min = x.min()
                self.max_val = 0.99*self.max_val+0.01*x_max
                self.min_val = 0.99*self.min_val+0.01*x_min
            n = 2 ** self.bit - 1
            scale = torch.clamp((self.max_val - self.min_val), min=1e-8) / float(n)
            zero_point = -self.min_val / scale
            zero_point = zero_point.round()
            scale = self._broadcast_qparams(scale)
            zero_point = self._broadcast_qparams(zero_point)
        return scale, zero_point

    def _forward(self, x_f, scale, zero_point):
        x_dq = STE.apply(x_f, scale, zero_point, self.qdesc, self.backend)
        return x_dq
