import operator
import torch
import torch.nn as nn
from sparsebit.quantization.modules import MultipleInputsQuantOpr, register_qmodule


@register_qmodule(sources=[operator.matmul, torch.matmul])
class MatMul(MultipleInputsQuantOpr):
    """量化矩阵乘法，但算子本身不包含量化 。

    量化输入在build_quantizer中处理, 通过在输入上增加QIdentity层来解决。
    """

    def __init__(self, org_module=None, config=None):
        super().__init__()
        self._repr_info = "QMatmul "

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor):
        out = torch.matmul(x_left, x_right)
        return out
