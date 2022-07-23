import torch
import torch.nn as nn
from sparsebit.quantization.common import get_backend
from sparsebit.quantization.quantizers import build_quantizer


class QuantOpr(nn.Module):
    def __init__(self):
        super(QuantOpr, self).__init__()
        self.weight = None
        self.input_quantizer = None
        self.weight_quantizer = None
        self.fake_fused = False  # a flag 用于表示该算子是否被fake_fused, 若fused则不进行量化

    def forward(self, x_in: torch.Tensor):
        raise NotImplementedError(
            "no found a forward in {}".format(self.__class__.__name__)
        )

    def build_quantizer(self, config):
        _backend = get_backend(config.BACKEND)
        if self.weight is not None:
            self.weight_quantizer = build_quantizer(cfg=config.W)
            self.weight_quantizer.set_backend(_backend)
        self.input_quantizer = build_quantizer(cfg=config.A)
        self.input_quantizer.set_backend(_backend)

    def set_fake_fused(self):
        self.fake_fused = True
        if self.weight_quantizer:
            self.weight_quantizer.set_fake_fused()
        if self.input_quantizer:
            self.input_quantizer.set_fake_fused()

    def set_quant(self, w_quant=False, a_quant=False):
        if self.weight_quantizer:
            if w_quant and not self.fake_fused:
                self.weight_quantizer.enable_quant()
            else:
                self.weight_quantizer.disable_quant()
        if self.input_quantizer:
            if a_quant and not self.fake_fused:
                self.input_quantizer.enable_quant()
            else:
                self.input_quantizer.disable_quant()

    def __repr__(self):
        info = self._repr_info + "fake_fused: {}".format(self.fake_fused)
        if self.weight_quantizer and self.weight_quantizer.is_enable:
            info += "\n\tweight_quantizer: {}".format(self.weight_quantizer.__repr__())
        if self.input_quantizer and self.input_quantizer.is_enable:
            info += "\n\tinput_quantizer: {}".format(self.input_quantizer.__repr__())

        return info
