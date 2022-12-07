import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsebit.quantization.modules import QuantOpr, register_qmodule


@register_qmodule(sources=[nn.Dropout])
class Dropout(nn.Module):
    def __init__(self, org_module, config=None):
        super().__init__()
        self.inplace = org_module.inplace
        self.p = org_module.p

    def forward(self, x_in):
        return F.dropout(x_in, self.p, training=self.training, inplace=self.inplace)


@register_qmodule(sources=[nn.Identity])
class QIdentity(QuantOpr):
    def __init__(self, org_module=None, config=None):
        super(QIdentity, self).__init__()
        self._repr_info = "QIdentity"

    def forward(self, x_in):
        x_in = self.input_quantizer(x_in)
        return x_in

@register_qmodule(sources=[torch.Tensor.float])
class Float(nn.Module):
    def __init__(self, org_module=None, config=None):
        super(Float, self).__init__()
        self._repr_info = "Float"

    def forward(self, x_in):
        return x_in.float()

@register_qmodule(sources=[torch.Tensor.bool])
class Bool(nn.Module):
    def __init__(self, org_module=None, config=None):
        super(Bool, self).__init__()
        self._repr_info = "Bool"

    def forward(self, x_in):
        return x_in.bool()

@register_qmodule(sources=[nn.Softmax, torch.Tensor.softmax, torch.softmax, F.softmax])
class QSoftmax(QuantOpr):
    def __init__(self, org_module=None, config=None):
        super().__init__()
        if isinstance(org_module, torch.fx.Node):
            if "dim" in org_module.kwargs:
                self.dim = org_module.kwargs["dim"]
            else:
                self.dim = org_module.args[1]
        elif isinstance(org_module, nn.Softmax):
            self.dim = org_module.dim

        self._repr_info = "QSoftmax "

    def forward(self, x_in, *args, **kwargs):
        if "dim" in kwargs:
            assert self.dim == kwargs["dim"], "parameter mismatch in softmax"
        elif len(args)>0:
            assert self.dim == args[0], "parameter mismatch in softmax"
        x_in = self.input_quantizer(x_in)
        out = F.softmax(x_in, dim=self.dim)
        return out


@register_qmodule(sources=[torch.Tensor.clone])
class Clone(nn.Module):
    """clone can be useful in quantization-aware training"""

    def __init__(self, org_module=None, config=None):
        super().__init__()

    def forward(self, x):
        return x.clone()

@register_qmodule(sources=[torch.zeros_like])
class Zeros_like(nn.Module):
    def __init__(self, org_module=None, config=None):
        super(Zeros_like, self).__init__()
        self._repr_info = "Zeros_like "

    def forward(self, x_in, *args):
        return torch.zeros_like(x_in, *args)

@register_qmodule(sources=[torch.Tensor.masked_fill])
class Masked_fill(nn.Module):
    def __init__(self, org_module=None, config=None):
        super().__init__()
        self._repr_info = "Masked_fill "

    def forward(self, x_in, *args, **kwargs):
        out = x_in.masked_fill(args[0], args[1])
        return out