import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsebit.quantization.modules import QuantOpr, register_qmodule


@register_qmodule(sources=[nn.Conv2d])
class QConv2d(QuantOpr):
    """
    Quantized Conv2d that can perform quantized convolution or normal convolution.
    """

    def __init__(self, org_module, config=None):
        assert isinstance(org_module, nn.Conv2d)
        super().__init__()
        self.cfg = config
        self.fwd_kwargs = dict(
            stride=org_module.stride,
            padding=org_module.padding,
            dilation=org_module.dilation,
            groups=org_module.groups,
        )
        self.weight = org_module.weight
        self.bias = org_module.bias
        self._repr_info = "Q" + org_module.__repr__()

    def forward(self, x_in: torch.Tensor):
        x_in = self.input_quantizer(x_in)
        weight = self.weight_quantizer(self.weight)
        out = F.conv2d(x_in, weight, self.bias, **self.fwd_kwargs)
        return out
