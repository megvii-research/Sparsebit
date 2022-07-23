import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsebit.quantization.modules import QuantOpr, register_qmodule


@register_qmodule(sources=[nn.Linear])
class QLinear(QuantOpr):
    def __init__(self, org_module, config=None):
        assert isinstance(org_module, nn.Linear)
        super().__init__()
        self.cfg = config
        self.weight = org_module.weight
        self.bias = org_module.bias
        self._repr_info = "Q" + org_module.__repr__()

    def forward(self, x_in: torch.Tensor):
        x_in = self.input_quantizer(x_in)
        weight = self.weight_quantizer(self.weight)
        out = F.linear(x_in, weight, self.bias)
        return out
