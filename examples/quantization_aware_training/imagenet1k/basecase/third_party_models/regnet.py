import os
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
import timm.models.layers as layers
from timm.models.regnet import _create_regnet

__all__ = [
    "regnetx_006",
]


def regnetx_006(pretrained=False, **kwargs):
    """RegNetX-600MF"""
    model = _create_regnet("regnetx_006", pretrained, **kwargs)
    model = _replace_BatchNormAct2d(model)
    model = _replace_Linear(model)
    return model


def _replace_BatchNormAct2d(model):
    finished = False
    while not finished:
        finished = _recurrency_replace_BatchNormAct2d(model)
    return model


def _recurrency_replace_BatchNormAct2d(module):
    finished = True
    for n, m in module.named_children():
        if isinstance(m, layers.BatchNormAct2d):
            setattr(module, n, BatchNormAct2d(m))
            finished = False
            break
        else:
            finished = _recurrency_replace_BatchNormAct2d(m)
            if not finished:
                break
    return finished


class BatchNormAct2d(nn.Module):
    def __init__(self, org_module=None):
        super().__init__()
        self.bn = nn.BatchNorm2d(org_module.weight.shape[0])
        self.bn.weight.data = org_module.weight.data
        self.bn.bias.data = org_module.bias.data
        self.bn.running_mean = org_module.running_mean
        self.bn.running_var = org_module.running_var
        self.act = org_module.act

    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        return x


def _replace_Linear(model):
    finished = False
    while not finished:
        finished = _recurrency_replace_Linear(model)
    return model


def _recurrency_replace_Linear(module):
    finished = True
    for n, m in module.named_children():
        if isinstance(m, layers.linear.Linear):
            setattr(module, n, Linear(m))
            finished = False
            break
        else:
            finished = _recurrency_replace_Linear(m)
            if not finished:
                break
    return finished


class Linear(nn.Module):
    def __init__(self, org_module=None):
        super().__init__()
        self.linear = nn.Linear(org_module.in_features, org_module.out_features)
        self.linear.weight = org_module.weight
        self.linear.bias = org_module.bias

    def forward(self, x):
        x = self.linear(x)
        return x
