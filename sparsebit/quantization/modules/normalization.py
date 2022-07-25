import torch
import torch.nn as nn
import torch.nn.functional as F
from . import QuantOpr, register_qmodule


@register_qmodule(sources=[nn.BatchNorm2d])
class QBatchNorm2d(QuantOpr):
    """量化BN层。由于默认为BN会跟在conv/linear前或后, 所以可以被fused, 故不执行量化。

    是QuantOpr的子类。
    """

    def __init__(self, org_module=None, config=None):
        super().__init__()
        self.module = org_module
        self._repr_info = "QBatchNorm2d "

    def forward(self, x_in):
        """BN层的前向传播,不做量化。"""
        out = self.module(x_in)
        return out
