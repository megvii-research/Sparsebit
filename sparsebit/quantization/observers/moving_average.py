import torch
from sparsebit.quantization.observers import Observer as BaseObserver
from sparsebit.quantization.observers import register_observer


@register_observer
class Observer(BaseObserver):
    TYPE = "moving_average"

    def __init__(self, config, qdesc):
        super(Observer, self).__init__(config, qdesc)
        self.ema_ratio = config.OBSERVER.MOVING_AVERAGE.EMA_RATIO

    def calc_minmax(self):
        assert self.qdesc.ch_axis>0, "Moving_average observer only support feature observing!"
        data = self.get_calibration_data(c_first=False)
        max_val, min_val = data[0].max(), data[0].min()
        for i in range(1, data.shape[0]):
            max_val = self.ema_ratio * max_val + (1 - self.ema_ratio) * data[i].max()
            min_val = self.ema_ratio * min_val + (1 - self.ema_ratio) * data[i].min()
        self.min_val = min_val.to(self.device)
        self.max_val = max_val.to(self.device)
        return self.min_val, self.max_val
