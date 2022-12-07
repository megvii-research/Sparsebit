import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsebit.quantization.modules import QuantOpr, register_qmodule


@register_qmodule(sources=[getattr])
class QGetAttr(nn.Module):
    def __init__(self, org_module=None, config=None):
        super(QGetAttr, self).__init__()
        assert isinstance(org_module, torch.fx.Node)
        self.target_attr = org_module.args[1]
        self._repr_info = "QGetAttr "

    def forward(self, x_in, *args):
        return getattr(x_in, self.target_attr)


@register_qmodule(sources=[operator.getitem])
class QGetItem(nn.Module):
    def __init__(self, org_module=None, config=None):
        super(QGetItem, self).__init__()
        self.target_item = org_module.args[1]
        self._repr_info = "QGetItem "

    def forward(self, x_in, *args):
        return x_in[self.target_item]


@register_qmodule(sources=[operator.eq])
class QEqual(nn.Module):
    def __init__(self, org_module=None, config=None):
        super(QEqual, self).__init__()
        self._repr_info = "QEqual "

    def forward(self, x_left, x_right):
        return x_left == x_right


@register_qmodule(sources=[operator.invert])
class Invert(nn.Module):
    def __init__(self, org_module=None, config=None):
        super(Invert, self).__init__()
        self._repr_info = "Invert "

    def forward(self, x_in):
        return ~x_in
