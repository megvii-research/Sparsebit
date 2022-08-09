import torch
import math
from sparsebit.quantization.observers import Observer as BaseObserver
from sparsebit.quantization.observers import register_observer


@register_observer
class Observer(BaseObserver):
    TYPE = "percentile"

    def __init__(self, config, qdesc):
        super(Observer, self).__init__(config, qdesc)
        self.alpha = config.OBSERVER.PERCENTILE.ALPHA

    def calc_minmax(self):
        data = self.get_calibration_data(c_first=True)
        channel = data.shape[0]
        if not self.is_perchannel:
            data = data.reshape(1, -1)
            channel = 1
        neg_length = (data<0).sum(-1)
        pos_length = (data>=0).sum(-1)

        max_val = torch.zeros(channel)
        min_val = torch.zeros(channel)
        for i in range(channel):
            max_val[i] = torch.kthvalue(
                    data[i], pos_length[i].item() + 1 - max(round(pos_length[i].item() * self.alpha), 1)
                ).values
            if neg_length[i]>0:
                min_val[i] = torch.kthvalue(
                    data[i],
                    max(round(neg_length[i].item() * self.alpha), 1),
                ).values

        self.min_val = min_val.to(self.device)
        self.max_val = max_val.to(self.device)
        return self.min_val, self.max_val
