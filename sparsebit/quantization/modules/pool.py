import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsebit.quantization.modules import QuantOpr, register_qmodule


@register_qmodule(sources=[nn.MaxPool2d])
class QMaxPool2d(QuantOpr):
    def __init__(self, org_module, config=None):
        super().__init__()
        self.fwd_kwargs = dict(
            kernel_size=org_module.kernel_size,
            stride=org_module.stride,
            padding=org_module.padding,
            dilation=org_module.dilation,
            ceil_mode=org_module.ceil_mode,
        )
        self._repr_info = "Q" + org_module.__repr__()

    def forward(self, x_in):
        # note: maxpool is non-quant default
        return F.max_pool2d(x_in, **self.fwd_kwargs)


@register_qmodule(sources=[nn.AdaptiveAvgPool2d, F.adaptive_avg_pool2d])
class QAdaptiveAvgPool2d(QuantOpr):
    def __init__(self, org_module, config=None):
        super().__init__()
        if isinstance(org_module, nn.Module):
            self.output_size = org_module.output_size
        else:
            self.output_size = org_module.args[1]
        self._repr_info = "Q" + org_module.__repr__()

    def forward(self, x_in, *args):
        x_in = self.input_quantizer(x_in)
        out = F.adaptive_avg_pool2d(x_in, self.output_size)
        return out
