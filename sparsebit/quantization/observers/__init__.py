OBSERVERS_MAP = {}


def register_observer(observer):
    OBSERVERS_MAP[observer.TYPE.lower()] = observer
    return observer


from .base import Observer
from . import minmax, percentile, mse, moving_average, kl_histogram, aciq


def build_observer(config, qdesc):
    observer = OBSERVERS_MAP[config.OBSERVER.TYPE.lower()](config, qdesc)
    return observer
