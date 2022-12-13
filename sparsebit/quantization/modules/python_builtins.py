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
        if self.target_attr != "shape":  # dynamic shape needs forward
            self.output = getattr(org_module.args[0], org_module.args[1])
        self._repr_info = "QGetAttr "

    def forward(self, x_in, *args):
        if self.target_attr == "shape":
            return x_in.shape()
        else:
            return self.output


@register_qmodule(sources=[operator.getitem])
class QGetItem(nn.Module):
    def __init__(self, org_module=None, config=None):
        super(QGetItem, self).__init__()
        self.target_item = org_module.args[1]
        self._repr_info = "QGetItem "

    def forward(self, x_in, *args):
        try:
            return x_in[self.target_item]
        except:
            import ipdb

            ipdb.set_trace()


@register_qmodule(sources=[operator.eq])
class QEqual(nn.Module):
    def __init__(self, org_module=None, config=None):
        super(QEqual, self).__init__()
        self._repr_info = "QEqual "

    def forward(self, x_left, x_right):
        return x_left == x_right
