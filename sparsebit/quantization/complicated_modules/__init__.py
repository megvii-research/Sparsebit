import torch.nn as nn
import timm.models.layers as layers


COMPLICATED_MODULE_MAP = {}


def register_complicated_module(sources: [nn.Module, str, ...]):
    def real_register(complicated_module):
        for src in sources:
            COMPLICATED_MODULE_MAP[src] = complicated_module
        return complicated_module

    return real_register


# 将需要注册的module文件填写至此
from .BatchNormAct2d import *
