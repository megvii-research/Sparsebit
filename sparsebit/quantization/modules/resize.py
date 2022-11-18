import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsebit.quantization.modules import QuantOpr, register_qmodule


@register_qmodule(sources=[nn.Upsample])
class QUpsample(nn.Module):
    def __init__(self, org_module=None, config=None):
        super(QUpsample, self).__init__()
        self.scale_factor = org_module.scale_factor
        self.mode = org_module.mode
        self._repr_info = "QUpsample"

    def forward(self, x_in, *args):
        out = F.interpolate(x_in, scale_factor=self.scale_factor, mode=self.mode)
        return out


@register_qmodule(sources=[F.interpolate])
class QInterpolate(nn.Module): # hack
    def __init__(self, org_module=None, config=None):
        super(QInterpolate, self).__init__()
        self._repr_info = "QInterpolate"
        if isinstance(org_module, nn.Module):
            raise NotImplementedError
        else:
            self.mode =org_module.kwargs["mode"]

    def forward(self, x_in, *args, **kwargs):
        out = F.interpolate(x_in, *args, **kwargs)
        return out
