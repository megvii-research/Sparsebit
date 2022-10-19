import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from sparsebit.quantization.modules import QuantOpr, register_qmodule


@register_qmodule(sources=[nn.ReLU, F.relu])
class QReLU(QuantOpr):
    """量化ReLU层,拥有 ``input_quantizer`` 。

    是QuantOpr的子类。

    Args:
        input_quantizer (sparsebit.quantization.quantizers.base.Quantizer):
            输入量化器。
        inplace (bool): 同 ``torch.nn.ReLU`` 。
    """

    def __init__(self, org_module, config=None):
        super().__init__()
        self._repr_info = "Q" + org_module.__repr__()
        if isinstance(org_module, nn.Module):
            self.inplace = org_module.inplace
        else:
            self.inplace = org_module.args[1]

    def forward(self, x_in):
        """ReLU层的前向传播,但加入了input量化。"""
        x_in = self.input_quantizer(x_in)
        out = F.relu(x_in, inplace=self.inplace)
        return out


@register_qmodule(sources=[nn.ReLU6, F.relu6])
class QReLU6(QuantOpr):
    """量化ReLU6层,拥有 ``input_quantizer`` 。

    是QuantOpr的子类。

    Args:
        input_quantizer (sparsebit.quantization.quantizers.base.Quantizer):
            输入量化器。
        inplace (bool): 同 ``torch.nn.ReLU6`` 。
    """

    def __init__(self, org_module, config=None):
        super().__init__()
        self._repr_info = "Q" + org_module.__repr__()
        if isinstance(org_module, nn.Module):
            inplace = org_module.inplace
        else:
            inplace = org_module.args[1]
        self.clamp = torch.clamp_ if inplace else torch.clamp

    def forward(self, x_in):
        """ReLU6层的前向传播,但加入了input量化。"""
        x_in = self.input_quantizer(x_in)
        out = self.clamp(x_in, min=0, max=6)
        return out


@register_qmodule(sources=[nn.LeakyReLU, F.leaky_relu])
class QLeakyReLU(QuantOpr):
    """量化LeakyReLU层,拥有 ``input_quantizer`` 。

    是QuantOpr的子类。

    Args:
        input_quantizer (sparsebit.quantization.quantizers.base.Quantizer):
            输入量化器。
        inplace (bool): 同 ``torch.nn.LeakyReLU`` 。
    """

    def __init__(self, org_module, config=None):
        super().__init__()
        self._repr_info = "Q" + org_module.__repr__()
        if isinstance(org_module, nn.Module):
            self.negative_slope = org_module.negative_slope
            self.inplace = org_module.inplace
        else:
            self.negative_slope = org_module.args[1]
            self.inplace = org_module.args[2]

    def forward(self, x_in):
        """LeakyReLU层的前向传播,但加入了input量化。"""
        x_in = self.input_quantizer(x_in)
        out = F.leaky_relu(
            x_in, negative_slope=self.negative_slope, inplace=self.inplace
        )
        return out


@register_qmodule(sources=[nn.Sigmoid, torch.sigmoid, F.sigmoid])
class QSigmoid(QuantOpr):
    """量化Sigmoid层,拥有 ``input_quantizer`` 。

    是QuantOpr的子类。

    Args:
        input_quantizer (sparsebit.quantization.quantizers.base.Quantizer):
            输入量化器。
    """

    def __init__(self, org_module, config=None):
        super().__init__()
        self._repr_info = "Q" + org_module.__repr__()

    def forward(self, x_in):
        """Sigmoid层的前向传播,但加入了input量化。"""
        x_in = self.input_quantizer(x_in)
        out = torch.sigmoid(x_in)
        return out


@register_qmodule(sources=[nn.SiLU, F.silu])
class QSiLU(QuantOpr):
    """量化SiLU层,拥有 ``input_quantizer`` 。

    是QuantOpr的子类。

    Args:
        input_quantizer (sparsebit.quantization.quantizers.base.Quantizer):
            输入量化器。
        inplace (bool): 同 ``torch.nn.SiLU`` 。
    """

    def __init__(self, org_module, config=None):
        super().__init__()
        self._repr_info = "Q" + org_module.__repr__()
        if isinstance(org_module, nn.Module):
            self.inplace = org_module.inplace
        else:
            self.inplace = org_module.args[1]

    def forward(self, x_in):
        """SiLU层的前向传播,但加入了input量化。"""
        x_in = self.input_quantizer(x_in)
        out = F.silu(x_in, inplace=self.inplace)
        return out


@register_qmodule(sources=[nn.GELU])
class QGELU(QuantOpr):
    def __init__(self, org_module, config=None):
        super().__init__()
        self._repr_info = "Q" + org_module.__repr__()

    def forward(self, x_in):
        x_in = self.input_quantizer(x_in)
        out = F.gelu(x_in)
        return out


@register_qmodule(sources=[nn.Mish, F.mish])
class QMish(QuantOpr):
    """量化Mish层,拥有 ``input_quantizer`` 。

    是QuantOpr的子类。

    Args:
        input_quantizer (sparsebit.quantization.quantizers.base.Quantizer):
            输入量化器。
        inplace (bool): 同 ``torch.nn.Mish`` 。
    """

    def __init__(self, org_module, config=None):
        super().__init__()
        self._repr_info = "Q" + org_module.__repr__()
        if isinstance(org_module, nn.Module):
            self.inplace = org_module.inplace
        else:
            self.inplace = org_module.args[1]

    def forward(self, x_in):
        """Mish层的前向传播,但加入了input量化。"""
        x_in = self.input_quantizer(x_in)
        out = F.mish(x_in, inplace=self.inplace)
        return out
