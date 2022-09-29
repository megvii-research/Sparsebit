import torch.nn as nn

SMODULE_MAP = {}


def register_smodule(sources: [nn.Module, str, ...]):
    def real_register(smodule):
        for src in sources:
            SMODULE_MAP[src] = smodule
        return smodule

    return real_register


# 将需要注册的module文件填写至此
from .base import SparseOpr
from .conv import *
from .linear import *
from .normalization import *
