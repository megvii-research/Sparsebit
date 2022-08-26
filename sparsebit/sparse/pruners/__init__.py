PRUNERS_MAP = {}


def register_pruner(pruner):
    PRUNERS_MAP[pruner.STRATEGY.lower()] = pruner
    return pruner


from .base import Pruner
from . import l1norm


def build_pruner(config):
    assert config.PRUNER.STRATEGY in PRUNERS_MAP, "no found an implement of {}".format(
        config.PRUNER.STRATEGY
    )
    pruner = PRUNERS_MAP[config.PRUNER.STRATEGY.lower()](config)
    return pruner
