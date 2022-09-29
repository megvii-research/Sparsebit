SPARSERS_MAP = {}


def register_sparser(sparser):
    SPARSERS_MAP[sparser.STRATEGY.lower()] = sparser
    return sparser


from .base import Sparser
from . import l1norm


def build_sparser(config, opr):
    assert (
        config.SPARSER.STRATEGY in SPARSERS_MAP
    ), "no found an implement of {}".format(config.SPARSER.STRATEGY)
    sparser = SPARSERS_MAP[config.SPARSER.STRATEGY.lower()](config, opr=opr)
    return sparser
