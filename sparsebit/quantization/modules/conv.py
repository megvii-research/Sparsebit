import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsebit.quantization.modules import QuantOpr, register_qmodule


@register_qmodule(sources=[nn.Conv2d])
class QConv2d(QuantOpr):
    """量化卷积层,拥有 ``input_quantizer`` 和 ``weight_quantizer`` 。

    是QuantOpr的子类。

    Attributes:
        input_quantizer (sparsebit.quantization.quantizers.base.Quantizer):
            输入量化器。
        weight_quantizer (sparsebit.quantization.quantizers.base.Quantizer):
            参数量化器。
        fwd_kwargs (Dict[str, any]): 运行 ``torch.nn.Conv2d`` forward需要的参数。
        weight (torch.nn.Parameter): 卷积层的weight,引用自原Module。
        bias (torch.nn.Parameter): 卷积层的bias,引用自原Module。
    """

    def __init__(self, org_module, config=None):
        assert isinstance(org_module, nn.Conv2d)
        super().__init__()
        self.cfg = config
        self.fwd_kwargs = dict(
            stride=org_module.stride,
            padding=org_module.padding,
            dilation=org_module.dilation,
            groups=org_module.groups,
        )
        self.weight = org_module.weight
        self.bias = org_module.bias
        self._repr_info = "Q" + org_module.__repr__()

    def forward(self, x_in: torch.Tensor):
        """卷积层的前向传播,但加入了input和weight量化。"""
        x_in = self.input_quantizer(x_in)
        weight = self.weight_quantizer(self.weight)
        out = F.conv2d(x_in, weight, self.bias, **self.fwd_kwargs)
        return out


@register_qmodule(sources=[nn.ConvTranspose2d])
class QConvTranspose2d(QuantOpr):
    def __init__(self, org_module, config=None):
        assert isinstance(org_module, nn.ConvTranspose2d)
        super().__init__()
        self.cfg = config
        self.fwd_kwargs = dict(
            stride=org_module.stride,
            padding=org_module.padding,
            output_padding=org_module.output_padding,
            dilation=org_module.dilation,
            groups=org_module.groups,
        )
        self.weight = org_module.weight
        self.bias = org_module.bias
        self._repr_info = "Q" + org_module.__repr__()

    def forward(self, x_in: torch.Tensor):
        """卷积层的前向传播,但加入了input和weight量化。"""
        x_in = self.input_quantizer(x_in)
        weight = self.weight_quantizer(self.weight)
        out = F.conv_transpose2d(x_in, weight, self.bias, **self.fwd_kwargs)
        return out
