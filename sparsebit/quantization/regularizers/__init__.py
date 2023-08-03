REGULARIZERS_MAP = {}


def register_regularizer(regularizer):
    REGULARIZERS_MAP[regularizer.TYPE.lower()] = regularizer
    return regularizer


from .base import Regularizer
from . import pact, bops, memory, bit_round


def build_regularizer(type, config, *args, **kwargs):
    regularizer = REGULARIZERS_MAP[type](config, *args, **kwargs)
    return regularizer