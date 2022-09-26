import torch.nn as nn

PMODULE_MAP = {}


def register_pmodule(sources: [nn.Module, str, ...]):
    def real_register(pmodule):
        for src in sources:
            PMODULE_MAP[src] = pmodule
        return pmodule

    return real_register


# 将需要注册的module文件填写至此
from .base import SparseOpr
from .conv import *
from .linear import *
from .normalization import *
