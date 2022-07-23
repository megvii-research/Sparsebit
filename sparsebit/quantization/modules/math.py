import operator
import torch
from sparsebit.quantization.modules import QuantOpr, register_qmodule


@register_qmodule(sources=[operator.add, torch.add])
class QAdd(QuantOpr):
    def __init__(self, org_module=None, config=None):
        super().__init__()
        self._repr_info = "QAdd"

    def forward(self, x_left, x_right):
        out = torch.add(x_left, x_right)
        return out


@register_qmodule(sources=[operator.mul, torch.mul])
class QMul(QuantOpr):
    def __init__(self, org_module=None, config=None):
        super().__init__()
        self._repr_info = "QMul"

    def forward(self, x_left, x_right):
        out = torch.mul(x_left, x_right)
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
