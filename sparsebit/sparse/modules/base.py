from abc import ABC

import torch
import torch.nn as nn
from sparsebit.sparse.sparsers import build_sparser


class SparseOpr(nn.Module, ABC):
    def __init__(self):
        super(SparseOpr, self).__init__()
        self._repr_info = "base"

    def forward(self, x_in: torch.Tensor):
        raise NotImplementedError(
            "no found a forward in {}".format(self.__class__.__name__)
        )

    def build_mask(self, pre_mask=None):
        raise NotImplementedError(
            "no found a calc_mask in {}".format(self.__class__.__name__)
        )

    def build_sparser(self, config):
        self.sparser = build_sparser(config, opr=self._repr_info)
