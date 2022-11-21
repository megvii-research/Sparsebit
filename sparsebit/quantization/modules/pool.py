import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsebit.quantization.modules import QuantOpr, register_qmodule


@register_qmodule(sources=[nn.MaxPool2d])
class MaxPool2d(nn.Module):
    """MaxPool层。认为maxpool不改变运算前后值域范围,所以不做量化。
    Attributes:
        fwd_kwargs (Dict[str, any]): 运行 ``torch.nn.functional.max_pool2d`` 需要的参数。
    """

    def __init__(self, org_module, config=None):
        super().__init__()
        self.fwd_kwargs = dict(
            kernel_size=org_module.kernel_size,
            stride=org_module.stride,
            padding=org_module.padding,
            dilation=org_module.dilation,
            ceil_mode=org_module.ceil_mode,
        )

    def forward(self, x_in):
        """MaxPool层的前向传播,不做量化。"""
        return F.max_pool2d(x_in, **self.fwd_kwargs)


@register_qmodule(sources=[nn.AdaptiveAvgPool2d, F.adaptive_avg_pool2d])
class QAdaptiveAvgPool2d(QuantOpr):
    """量化AvgPool层,拥有 ``input_quantizer`` 。

    是QuantOpr的子类。

    Attributes:
        input_quantizer (sparsebit.quantization.quantizers.base.Quantizer):
            输入量化器。
        output_size (any):
            同 ``torch.nn.AdaptiveAvgPool2d`` 。
    """

    def __init__(self, org_module, config=None):
        super().__init__()
        if isinstance(org_module, nn.Module):
            self.output_size = org_module.output_size
        else:
            self.output_size = org_module.args[1]
        self._repr_info = "Q" + org_module.__repr__()

    def forward(self, x_in, *args):
        """AvgPool层的前向传播,但加入了input量化。"""
        x_in = self.input_quantizer(x_in)
        out = F.adaptive_avg_pool2d(x_in, self.output_size)
        return out
