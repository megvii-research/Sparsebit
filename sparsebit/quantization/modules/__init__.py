import torch.nn as nn
from sparsebit.utils import check_torch_version

QMODULE_MAP = {}


def register_qmodule(sources: [nn.Module, str, ...]):
    def real_register(qmodule):
        for src in sources:
            QMODULE_MAP[src] = qmodule
        return qmodule

    return real_register


# 将需要注册的module文件填写至此
from .base import QuantOpr, MultipleInputsQuantOpr
from .activations import *
from .conv import *
from .linear import *
from .math import *
from .pool import *
from .shape import *
from .normalization import *
from .unary import *
from .resize import *
from .matmul import *
from .python_builtins import *
from .embedding import *

if check_torch_version("1.11.0"):
    from .torchvision_ops import *


PASSTHROUGHT_MODULES = (
    QAdd,
    QSubtract,
    QMul,
    QDivide,
    QFloorDiv,
    QBatchNorm2d,
    QLayerNorm,
    QIdentity,
    Concat,
    QGetAttr,
    QGetItem,
    QEqual,
    Size,
    Transpose,
    Reshape,
    Permute,
    Expand,
)
