import torch
import torch.nn as nn
from sparsebit.quantization.quantizers import Quantizer as BaseQuantizer
from sparsebit.quantization.quantizers import register_quantizer
from .quant_tensor import STE


@register_quantizer
class Quantizer(BaseQuantizer):
    TYPE = "uniform_symmetric_qat"

    def __init__(self, config):
        super(Quantizer, self).__init__(config)
        self.init_params = False  # uniform_qat需要基于calibration做初始化

    def calc_qparams(self, bit_allocation):
        if self.fake_fused:
            return self.scale, self.zero_point
        scale, _ = self.observer.calc_qparams(bit_allocation)
        self.zero_point = self._broadcast_qparams(torch.zeros_like(scale))
        self.max_val = self.observer.max_val.to(self.device)
        self.min_val = self.observer.min_val.to(self.device)
        self.init_params = True
        return self.scale, self.zero_point

    def _qparams_preprocess(self, x):
        with torch.no_grad():
            n = 2 ** (self.bit - 1) - 1
            scale, _ = torch.max(torch.stack([self.min_val.abs(), self.max_val.abs()], dim=1), dim=1)
            scale = torch.clamp(scale, min=1e-8) / n
            scale = self._broadcast_qparams(scale)
        return scale, self.zero_point

    def _forward(self, x_f, scale, zero_point):
        x_dq = STE.apply(x_f, scale, zero_point, self.qdesc, self.backend)
        return x_dq
