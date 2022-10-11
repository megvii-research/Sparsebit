from abc import ABC

from torch import nn


class Sparser(nn.Module, ABC):
    def __init__(self, config, opr):
        super(Sparser, self).__init__()
        self.config = config
        self.opr = opr
        self.type = config.SPARSER.TYPE
        self.strategy = config.SPARSER.STRATEGY
        self.ratio = config.SPARSER.RATIO

    def calc_mask(self, x):
        pass

    def set_ratio(self, ratio):
        self.ratio = ratio

    def __repr__(self):
        info = "{}, {}, {}".format(self.type, self.strategy, self.ratio)

        return info
