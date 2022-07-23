import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from sparsebit.quantization.modules import QuantOpr, register_qmodule


@register_qmodule(sources=[nn.ReLU, F.relu])
class QReLU(QuantOpr):
    def __init__(self, org_module, config=None):
        super().__init__()
        self._repr_info = "Q" + org_module.__repr__()
        if isinstance(org_module, nn.Module):
            self.inplace = org_module.inplace
        else:
            self.inplace = org_module.args[1]

    def forward(self, x_in):
        x_in = self.input_quantizer(x_in)
        out = F.relu(x_in, inplace=self.inplace)
        return out


@register_qmodule(sources=[nn.ReLU6, F.relu6])
class QReLU6(QuantOpr):
    def __init__(self, org_module, config=None):
        super().__init__()
        self._repr_info = "Q" + org_module.__repr__()
        if isinstance(org_module, nn.Module):
            inplace = org_module.inplace
        else:
            inplace = org_module.args[1]
        self.clamp = torch.clamp_ if inplace else torch.clamp

    def forward(self, x_in):
        x_in = self.input_quantizer(x_in)
        out = self.clamp(x_in, min=0, max=6)
        return out


@register_qmodule(sources=[nn.Sigmoid, torch.sigmoid, F.sigmoid])
class QSigmoid(QuantOpr):
    def __init__(self, org_module, config=None):
        super().__init__()
        self._repr_info = "Q" + org_module.__repr__()

    def forward(self, x_in):
        x_in = self.input_quantizer(x_in)
        out = torch.sigmoid(x_in)
        return out


@register_qmodule(sources=[nn.SiLU, F.silu])
class QSiLU(QuantOpr):
    def __init__(self, org_module, config=None):
        super().__init__()
        self._repr_info = "Q" + org_module.__repr__()
        if isinstance(org_module, nn.Module):
            self.inplace = org_module.inplace
        else:
            self.inplace = org_module.args[1]

    def forward(self, x_in):
        x_in = self.input_quantizer(x_in)
        out = F.silu(x_in, inplace=self.inplace)
        return out
