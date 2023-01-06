import torch
from torch import nn
from torch.quantization.observer import _ObserverBase
from sparsebit.quantization.common import Granularity, QuantTarget


class DataCache(object):
    def __init__(self, qdesc):
        self.qdesc = qdesc
        self._data_cache = []

    def update(self, data):
        self._data_cache.append(data)

    def reset(self):
        self._data_cache = []

    def __len__(self):
        return len(self._data_cache)

    def get_data_for_calibration(self, granularity: Granularity):
        assert len(self._data_cache), "No data cached!"
        assert granularity in [
            Granularity.LAYERWISE,
            Granularity.CHANNELWISE,
        ], "only layerwise or channelwise quantization are supported now!"
        if granularity == Granularity.CHANNELWISE:
            data = torch.cat(self._data_cache, dim=self.qdesc.ch_axis)
            if self.qdesc.ch_axis != 0:
                data = data.transpose(0, self.qdesc.ch_axis)
            data = data.flatten(1)
        elif granularity == Granularity.LAYERWISE:
            data = torch.cat([d.reshape(-1) for d in self._data_cache], axis=0)
        else:
            raise NotImplementedError
        return data

    def get_batch_size(self):
        if self.qdesc.target == QuantTarget.WEIGHT:
            return None
        return sum([d.shape[self.qdesc.bs_axis] for d in self._data_cache])

    def get_data_cache(self):
        assert len(self._data_cache), "No data cached!"
        return self._data_cache


class Observer(nn.Module):
    def __init__(self, config, qdesc):
        super(Observer, self).__init__()
        self.cfg = config
        self.qdesc = qdesc
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.register_buffer("min_val", torch.tensor(float("-inf")).to(self.device))
        self.register_buffer("max_val", torch.tensor(float("inf")).to(self.device))
        self.data_cache = DataCache(qdesc)

    def calc_qparams(self, bit_allocation):
        if bit_allocation:
            assert (
                len(self.cfg.OBSERVER.BIT_CHOICES) > 0
            ), "Please assign bit choices before applying bit allocation!"
            self.scales, self.zero_points = {}, {}
            for bit in self.cfg.OBSERVER.BIT_CHOICES:
                min_val, max_val = self.calc_minmax(bit)
                scale, zero_point = self.calc_qparams_with_minmax(min_val, max_val, bit)
                self.scales[bit] = scale
                self.zero_points[bit] = zero_point

        min_val, max_val = self.calc_minmax(self.qdesc.bit)
        scale, zero_point = self.calc_qparams_with_minmax(
            min_val, max_val, self.qdesc.bit
        )
        self.data_cache.reset()
        assert len(self.data_cache) == 0, "free data cache after calc_qparams"
        return scale, zero_point

    def calc_qparams_with_minmax(self, min_val, max_val, bit):
        min_val_neg = torch.minimum(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.maximum(max_val, torch.zeros_like(max_val))
        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.float32, device=device)
        qmin, qmax, _ = self.qdesc.calc_qmin_qmax(bit, self.qdesc._scheme)
        if self.is_symmetric:
            max_val_pos = torch.maximum(-min_val_neg, max_val_pos)
            scale = max_val_pos * 2 / float(qmax - qmin)
            scale = torch.maximum(scale, torch.tensor(1e-6))
        else:
            scale = (max_val_pos - min_val_neg) / float(qmax - qmin)
            scale = torch.maximum(scale, torch.tensor(1e-6))
            zero_point = torch.round(-min_val_neg / scale)
        return scale, zero_point

    @property
    def is_perchannel(self):
        return self.qdesc.is_perchannel

    @property
    def is_symmetric(self):
        return self.qdesc.is_symmetric
