from abc import ABC


class Pruner(ABC):
    def __init__(self, config):
        self.config = config

    def calc_mask(self, x):
        pass
