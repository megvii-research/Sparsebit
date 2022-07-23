import torch
import torch.nn as nn
import torch.nn.functional as F
from . import QuantOpr, register_qmodule


@register_qmodule(sources=[nn.BatchNorm2d])
class QBatchNorm2d(QuantOpr):
    def __init__(self, org_module=None, config=None):
        super().__init__()
        self.module = org_module
        self._repr_info = "QBatchNorm2d "

    def forward(self, x_in):
        out = self.module(x_in)
        return out
