import torch
from sparsebit.quantization.observers import Observer as BaseObserver
from sparsebit.quantization.observers import register_observer


@register_observer
class Observer(BaseObserver):
    TYPE = "minmax"

    def __init__(self, config, qdesc):
        super(Observer, self).__init__(config, qdesc)

    def calc_minmax(self):
        assert (
            len(self.data_cache) > 0
        ), "Before calculating the quant params, the observation of data should be done"
        if self.qdesc.ch_axis > 0:
            data = torch.cat(self.data_cache, axis=0)
            data = (
                data.transpose(self.qdesc.ch_axis, 0)
                .reshape(data.shape[self.qdesc.ch_axis], -1)
                .detach()
                .data
            )
        else:
            data = torch.cat(self.data_cache, axis=1)
            data = data.reshape(data.shape[self.qdesc.ch_axis], -1).detach().data
        self.reset_data_cache()
        if self.is_perchannel:
            max_val = data.max(axis=1).values
            min_val = data.min(axis=1).values
        else:
            min_val, max_val = data.min(), data.max()
        self.min_val = min_val.to(self.device)
        self.max_val = max_val.to(self.device)
        return self.min_val, self.max_val
