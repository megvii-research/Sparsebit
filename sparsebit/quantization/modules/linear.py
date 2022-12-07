import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsebit.quantization.modules import QuantOpr, register_qmodule


@register_qmodule(sources=[nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear])
class QLinear(QuantOpr):
    """量化全连接层,拥有 ``input_quantizer`` 和 ``weight_quantizer`` 。

    是QuantOpr的子类。

    Attributes:
        input_quantizer (sparsebit.quantization.quantizers.base.Quantizer):
            输入量化器。
        weight_quantizer (sparsebit.quantization.quantizers.base.Quantizer):
            参数量化器。
        weight (torch.nn.Parameter): 卷积层的weight,引用自原Module。
        bias (torch.nn.Parameter): 卷积层的bias,引用自原Module。
    """

    def __init__(self, org_module, config=None):
        assert isinstance(org_module, nn.Linear)
        super().__init__()
        self.cfg = config
        self.weight = org_module.weight
        self.bias = org_module.bias
        self._repr_info = "Q" + org_module.__repr__()

    def forward(self, x_in: torch.Tensor):
        """全连接层的前向传播,但加入了input和weight量化。"""
        x_in = self.input_quantizer(x_in)
        weight = self.weight_quantizer(self.weight)
        out = F.linear(x_in, weight, self.bias)
        return out
