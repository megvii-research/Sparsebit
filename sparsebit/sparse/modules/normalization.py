import torch
import torch.nn as nn
import torch.nn.functional as F
from . import SparseOpr, register_smodule


@register_smodule(sources=[nn.BatchNorm2d])
class SBatchNorm2d(SparseOpr):
    def __init__(self, org_module, config=None):
        super().__init__()
        self.module = org_module
        self.mask = torch.ones([1, self.module.num_features, 1, 1])
        self._repr_info = "S" + org_module.__repr__()

    def calc_mask(self, pre_mask=None):
        if (
            self.sparser.type == "structed"
            and self.sparser.strategy == "l1norm"
            and pre_mask is not None
        ):
            pre_mask = pre_mask.reshape(self.mask.shape)
            self.mask.data.copy_(pre_mask.data)

        return None

    def forward(self, x_in):
        out = self.module(x_in) * self.mask.to(x_in.device)
        return out
