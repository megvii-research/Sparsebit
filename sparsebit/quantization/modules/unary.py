import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsebit.quantization.modules import QuantOpr, register_qmodule


@register_qmodule(sources=[nn.Dropout])
class QDropout(QuantOpr):
    def __init__(self, org_module, config=None):
        super().__init__()
        self.inplace = org_module.inplace
        self.p = org_module.p
        self._repr_info = "Q" + org_module.__repr__()

    def forward(self, x_in):
        out = F.dropout(x_in, self.p, inplace=self.inplace)
        return out


@register_qmodule(sources=[nn.Identity])
class QIdentity(QuantOpr):
    def __init__(self, org_module=None, config=None):
        super(QIdentity, self).__init__()
        self._repr_info = "QIdentity"

    def forward(self, x_in):
        return x_in
