import operator
import torch
import torch.nn as nn
from sparsebit.quantization.modules import QuantOpr, register_qmodule


@register_qmodule(sources=[operator.add, torch.add])
class QAdd(nn.Module):
    def __init__(self, org_module=None, config=None):
        super().__init__()
        self._repr_info = "QAdd"

    def forward(self, x_left, x_right):
        out = torch.add(x_left, x_right)
        return out


@register_qmodule(sources=[operator.sub, torch.subtract])
class QSubtract(nn.Module):
    def __init__(self, org_module=None, config=None):
        super().__init__()
        self._repr_info = "QSubtract "

    def forward(self, x_left, x_right):
        out = torch.subtract(x_left, x_right)
        return out


@register_qmodule(sources=[operator.mul, torch.mul])
class QMul(nn.Module):
    def __init__(self, org_module=None, config=None):
        super().__init__()
        self._repr_info = "QMul"

    def forward(self, x_left, x_right):
        out = torch.mul(x_left, x_right)
        return out


@register_qmodule(sources=[operator.truediv, torch.div])
class QDivide(nn.Module):
    def __init__(self, org_module=None, config=None):
        super().__init__()
        self._repr_info = "QDivide "

    def forward(self, x_left, x_right):
        out = torch.divide(x_left, x_right)  # x / y
        return out


@register_qmodule(sources=[operator.floordiv, torch.floor_divide])
class QFloorDiv(nn.Module):
    def __init__(self, org_module=None, config=None):
        super().__init__()
        self._repr_info = "QFloorDiv "

    def forward(self, x_left, x_right):
        out = torch.floor_divide(x_left, x_right)  # x // y
        return out


@register_qmodule(sources=[torch.mean, torch.Tensor.mean, "mean"])
class QMean(QuantOpr):
    def __init__(self, org_module=None, config=None):
        super(QMean, self).__init__()
        self._repr_info = "QMean"
        if isinstance(org_module, torch.fx.Node):
            self.dim = org_module.args[1]
            self.keepdim = org_module.kwargs["keepdim"]
        else:
            raise NotImplementedError

    def forward(self, x_in, *args, **kwargs):
        x_in = self.input_quantizer(x_in)
        out = torch.mean(x_in, dim=self.dim, keepdim=self.keepdim)
        return out

@register_qmodule(sources=[torch.sum,torch.Tensor.sum])
class QSum(QuantOpr):
    def __init__(self, org_module=None, config=None):
        super(QSum, self).__init__()
        self._repr_info = "QSum "

    def forward(self, x_in, *args, **kwargs):
        x_in = self.input_quantizer(x_in)
        return x_in.sum(*args, **kwargs)

@register_qmodule(sources=[torch.cumsum,torch.Tensor.cumsum])
class QCumsum(QuantOpr):
    def __init__(self, org_module=None, config=None):
        super(QCumsum, self).__init__()
        self._repr_info = "QCumsum "

    def forward(self, x_in, *args, **kwargs):
        x_in = self.input_quantizer(x_in)
        return x_in.cumsum(*args, **kwargs)

@register_qmodule(sources=[torch.sin,torch.Tensor.sin])
class Sin(nn.Module):
    def __init__(self, org_module=None, config=None):
        super(Sin, self).__init__()
        self._repr_info = "Sin "

    def forward(self, x_in):
        return x_in.sin()

@register_qmodule(sources=[torch.cos,torch.Tensor.cos])
class Cos(nn.Module):
    def __init__(self, org_module=None, config=None):
        super(Cos, self).__init__()
        self._repr_info = "Cos "

    def forward(self, x_in):
        return x_in.cos()