from abc import ABC


class Sparser(ABC):
    def __init__(self, config):
        self.config = config

    def calc_mask(self, x):
        pass
