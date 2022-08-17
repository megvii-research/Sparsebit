import torch
from torch import nn
from torch.quantization.observer import _ObserverBase


class Observer(nn.Module):
    def __init__(self, config, qdesc):
        super(Observer, self).__init__()
        self.cfg = config
        self.qdesc = qdesc
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.register_buffer("min_val", torch.tensor(float("-inf")).to(self.device))
        self.register_buffer("max_val", torch.tensor(float("inf")).to(self.device))
        self.reset_data_cache()

    def calc_qparams(self):
        qmin, qmax = self.qdesc.qrange
        min_val, max_val = self.calc_minmax()
        min_val_neg = torch.minimum(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.maximum(max_val, torch.zeros_like(max_val))
        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.float32, device=device)
        if self.is_symmetric:
            max_val_pos = torch.maximum(-min_val_neg, max_val_pos)
            scale = max_val_pos * 2 / float(qmax - qmin)
            scale = torch.maximum(scale, torch.tensor(1e-6))
        else:
            scale = (max_val_pos - min_val_neg) / float(qmax - qmin)
            scale = torch.maximum(scale, torch.tensor(1e-6))
            zero_point = torch.round(-min_val_neg / scale)
        assert len(self._data_cache) == 0, "free data cache after calc_qparams"
        return scale, zero_point

    def get_calibration_data(self, c_first=False):
        assert (
            len(self._data_cache) > 0
        ), "Before calculating the quant params, the observation of data should be done"

        if c_first:
            if self.qdesc.ch_axis > 0:
                data = torch.cat(self._data_cache, axis=0)
                data = (
                    data.transpose(self.qdesc.ch_axis, 0)
                    .reshape(data.shape[self.qdesc.ch_axis], -1)
                    .detach()
                    .data
                )
            else:
                data = torch.cat(self._data_cache, axis=1)
                data = data.reshape(data.shape[self.qdesc.ch_axis], -1).detach().data
        else:
            data = torch.cat(self._data_cache, axis=0)
        self.reset_data_cache()
        return data

    def update(self, data):
        self._data_cache.append(data)

    def reset_data_cache(self):
        self._data_cache = []

    @property
    def is_perchannel(self):
        return self.qdesc.is_perchannel

    @property
    def is_symmetric(self):
        return self.qdesc.is_symmetric
