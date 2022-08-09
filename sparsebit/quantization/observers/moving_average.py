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
        data = self.get_calibration_data(c_first=True)
        max_val, min_val = None, None
        for d in data:
            if self.is_perchannel:
                if max_val:
                    max_val = (
                        self.ema_ratio * max_val
                        + (1 - self.ema_ratio) * d.max(axis=1).values
                    )
                    min_val = (
                        self.ema_ratio * min_val
                        + (1 - self.ema_ratio) * d.min(axis=1).values
                    )
                else:
                    max_val = d.max(axis=1).values
                    min_val = d.min(axis=1).values
            else:
                if max_val:
                    max_val = self.ema_ratio * max_val + (1 - self.ema_ratio) * d.max()
                    min_val = self.ema_ratio * min_val + (1 - self.ema_ratio) * d.min()
                else:
                    min_val, max_val = d.min(), d.max()
        self.min_val = min_val.to(self.device)
        self.max_val = max_val.to(self.device)
        return self.min_val, self.max_val

    def get_calibration_data(self, c_first=False):
        assert (
            len(self._data_cache) > 0
        ), "Before calculating the quant params, the observation of data should be done"
        if c_first:
            data = []
            if self.qdesc.ch_axis > 0:
                for d in self._data_cache:
                    d = (
                        d.transpose(self.qdesc.ch_axis, 0)
                        .reshape(d.shape[self.qdesc.ch_axis], -1)
                        .detach()
                        .data
                    )
                    data.append(d)

            else:
                for d in self._data_cache:
                    d = d.reshape(d.shape[self.qdesc.ch_axis], -1).detach().data
                    data.append(d)
        else:
            data = torch.cat(self._data_cache, axis=0)
        self.reset_data_cache()
        return data
