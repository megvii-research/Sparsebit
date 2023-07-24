import torch
from sparsebit.quantization.observers import Observer as BaseObserver
from sparsebit.quantization.observers import register_observer
from sparsebit.quantization.common import Granularity, QuantTarget


@register_observer
class Observer(BaseObserver):
    TYPE = "minmax"

    def __init__(self, config, qdesc):
        super(Observer, self).__init__(config, qdesc)

    def calc_minmax(self):
        data = self.data_cache.get_data_for_calibration(self.granularity)
        if self.granularity in [Granularity.CHANNELWISE, Granularity.GROUPWISE]:
            max_val = data.max(axis=1).values
            min_val = data.min(axis=1).values
        else:  # Granularity.LAYERWISE
            min_val, max_val = data.min(), data.max()

        self.data_cache.reset()
        self.min_val = min_val.to(self.device)
        self.max_val = max_val.to(self.device)
        return self.min_val, self.max_val
