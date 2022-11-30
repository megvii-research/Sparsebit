import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsebit.sparse.modules import SparseOpr, register_smodule


@register_smodule(sources=[nn.Linear])
class SLinear(SparseOpr):
    def __init__(self, org_module, config=None):
        assert isinstance(org_module, nn.Linear)
        super().__init__()
        self.weight = org_module.weight
        self.bias = org_module.bias
        w_mask = torch.ones_like(self.weight)
        b_mask = torch.ones_like(self.bias) if self.bias is not None else None
        self.register_buffer("w_mask", w_mask)
        self.register_buffer("b_mask", b_mask)
        self._repr_info = "S" + org_module.__repr__()

    def calc_mask(self, pre_mask=None):
        self.w_mask = self.sparser.calc_mask(self.weight)

        if self.sparser.type == "structed":
            mask = self.w_mask[:, 0]
            if self.bias is not None:
                self.b_mask.data.copy_(mask.data)

        return None

    def forward(self, x_in: torch.Tensor):
        weight = self.weight * self.w_mask
        bias = self.bias * self.b_mask if self.bias is not None else self.bias
        out = F.linear(x_in, weight, bias)
        return out
