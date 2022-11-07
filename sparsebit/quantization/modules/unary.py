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
        return F.dropout(x_in, self.p, training=self.training, inplace=self.inplace)


@register_qmodule(sources=[nn.Identity])
class QIdentity(QuantOpr):
    def __init__(self, org_module=None, config=None):
        super(QIdentity, self).__init__()
        self._repr_info = "QIdentity"

    def forward(self, x_in):
        x_in = self.input_quantizer(x_in)
        return x_in


@register_qmodule(sources=[nn.Softmax, torch.Tensor.softmax, F.softmax])
class Softmax(QuantOpr):
    def __init__(self, org_module=None, config=None):
        super().__init__()
        assert isinstance(org_module, torch.fx.Node)
        if "dim" in org_module.kwargs:
            self.dim = org_module.kwargs["dim"]
        else:
            self.dim = org_module.args[1]

        self._repr_info = "QSoftmax "

    def forward(self, x_in, *args, **kwargs):
        if "dim" in kwargs:
            assert self.dim == kwargs["dim"], "parameter mismatch in softmax"
        else:
            assert self.dim == args[0], "parameter mismatch in softmax"
        out = F.softmax(x_in, dim=self.dim)
        return out


@register_qmodule(sources=[torch.Tensor.clone])
class Clone(nn.Module):
    """clone can be useful in quantization-aware training"""

    def __init__(self, org_module=None, config=None):
        super().__init__()

    def forward(self, x):
        return x.clone()
