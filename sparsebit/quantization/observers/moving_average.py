import torch
from sparsebit.quantization.observers import Observer as BaseObserver
from sparsebit.quantization.observers import register_observer
from sparsebit.quantization.common import Granularity, QuantTarget


@register_observer
class Observer(BaseObserver):
    TYPE = "moving_average"

    def __init__(self, config, qdesc):
        super(Observer, self).__init__(config, qdesc)
        assert (
            hasattr(config.OBSERVER, "MOVING_AVERAGE")
            and self.qdesc.target == QuantTarget.FEATURE
        ), "Moving_average observer only support feature observing!"
        assert (
            self.granularity == Granularity.LAYERWISE
        ), "Moving_average observer only support layerwise quantization!"
        self.ema_ratio = config.OBSERVER.MOVING_AVERAGE.EMA_RATIO

    def calc_minmax(self):
        data = self.data_cache.get_data_cache()
        self.data_cache.reset()
        max_val, min_val = None, None
        for data_batch in data:
            if self.qdesc.bs_axis > 0:
                data_batch = data_batch.transpose(0, self.qdesc.bs_axis)
            for d in data_batch:
                if max_val == None and min_val == None:
                    max_val, min_val = d.max(), d.min()
                else:
                    max_val = self.ema_ratio * max_val + (1 - self.ema_ratio) * d.max()
                    min_val = self.ema_ratio * min_val + (1 - self.ema_ratio) * d.min()
        self.min_val = min_val.to(self.device)
        self.max_val = max_val.to(self.device)
        return self.min_val, self.max_val
