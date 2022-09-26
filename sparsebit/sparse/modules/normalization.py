import torch
import torch.nn as nn
import torch.nn.functional as F
from . import SparseOpr, register_pmodule


@register_pmodule(sources=[nn.BatchNorm2d])
class PBatchNorm2d(SparseOpr):
    def __init__(self, org_module, config=None):
        super().__init__()
        self.module = org_module
        self.mask = torch.ones([1, self.module.num_features, 1, 1])
        self._repr_info = "PBatchNorm2d"

    def calc_mask(self):
        pass

    def forward(self, x_in):
        out = self.module(x_in) * self.mask.to(x_in.device)
        return out
