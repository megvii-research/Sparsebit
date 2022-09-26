from abc import ABC


class Sparser(ABC):
    def __init__(self, config, opr):
        self.config = config
        self.opr = opr

    def calc_mask(self, x):
        pass
