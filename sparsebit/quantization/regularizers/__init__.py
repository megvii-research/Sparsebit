REGULARIZERS_MAP = {}


def register_regularizer(regularizer):
    REGULARIZERS_MAP[regularizer.TYPE.lower()] = regularizer
    return regularizer


from .base import Regularizer
from . import dampen


def build_regularizer(config):
    regularizer = REGULARIZERS_MAP[config.REGULARIZER.TYPE.lower()](config)
    return regularizer