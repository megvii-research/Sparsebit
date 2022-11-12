import torch
import numpy as np
from sparsebit.quantization.quantizers import Quantizer as BaseQuantizer
from sparsebit.quantization.quantizers import register_quantizer
from .quant_tensor import STE


@register_quantizer
class Quantizer(BaseQuantizer):
    TYPE = "DoReFa"

    def __init__(self, config):
        super(Quantizer, self).__init__(config)

    def _forward(self, x, scale, zero_point):
        x_tanhed = x.tanh()
        x_normed = x_tanhed / x_tanhed.detach().abs().max()  # norm to [-1, +1]
        scale, zero_point = self.scale, self.zero_point
        x_normed_fq = STE.apply(x_normed, scale, zero_point, self.qdesc, self.backend)
        return x_normed_fq

    def update_observer(self, x):
        self.dims = len(x.shape)
        x_tanhed = x.detach().tanh()
        x_normed = x_tanhed / x_tanhed.detach().abs().max()
        self.observer.data_cache.update(x_normed)
