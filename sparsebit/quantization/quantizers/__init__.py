QUANTIZERS_MAP = {}


def register_quantizer(quantizer):
    QUANTIZERS_MAP[quantizer.TYPE.lower()] = quantizer
    return quantizer


from .base import Quantizer
from . import uniform
from . import lsq
from . import dorefa
from . import lsq_plus
from . import pact
from . import adaround


def build_quantizer(cfg):
    assert cfg.QUANTIZER.TYPE in QUANTIZERS_MAP, "no found an implement of {}".format(
        cfg.QUANTIZER.TYPE
    )
    quantizer = QUANTIZERS_MAP[cfg.QUANTIZER.TYPE.lower()](cfg)
    return quantizer
