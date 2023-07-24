import torch
from torch import nn
from torch.quantization.observer import _ObserverBase
from sparsebit.quantization.common import Granularity, QuantTarget


class DataCache(object):
    def __init__(self, qdesc):
        self.qdesc = qdesc
        self._data_cache = []

    def update(self, data):
        if self.ch_axis != 0:
            self._data_cache.append(data.transpose(self.ch_axis, 0))
        else:
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
            Granularity.GROUPWISE,
        ], "only layerwise, channelwise and groupwise quantization are supported now!"
        if granularity == Granularity.LAYERWISE:
            data = torch.cat([d.reshape(1, -1) for d in self._data_cache], axis=1)
        elif granularity == Granularity.CHANNELWISE:
            data = torch.cat(
                [d.reshape(d.shape[0], -1) for d in self._data_cache], axis=1
            )
        elif granularity == Granularity.GROUPWISE:
            if self.target == QuantTarget.FEATURE:  # feature group on channel dim
                assert (
                    self._data_cache[0].shape[0] <= self.group_size
                    or self._data_cache[0].shape[0] % self.group_size == 0
                ), "group size must be divided by channel num! got {} and {} instead".format(
                    self.group_size, self._data_cache[0].shape[0]
                )
                group_num = max(self._data_cache[0].shape[0] // self.group_size, 1)
                if group_num == 1:
                    self.qdesc.set_group_size = self._data_cache[0].shape[0]
                data = torch.cat(
                    [d.reshape(group_num, -1) for d in self._data_cache], axis=1
                )
            else:  # weight group on ic dim
                assert (
                    self._data_cache[0].shape[1] <= self.group_size
                    or self._data_cache[0].shape[1] % self.group_size == 0
                ), "group size must be divided by ic num! got {} and {} instead".format(
                    self.group_size, self._data_cache[0].shape[1]
                )
                group_num = max(self._data_cache[0].shape[1] // self.group_size, 1)
                if group_num == 1:
                    self.qdesc.set_group_size = self._data_cache[0].shape[1]
                data = torch.cat(
                    [d.reshape(d.shape[0] * group_num, -1) for d in self._data_cache],
                    axis=1,
                )
        return data

    def get_batch_size(self):
        if self.qdesc.target == QuantTarget.WEIGHT:
            return None
        return sum([d.shape[self.qdesc.bs_axis] for d in self._data_cache])

    def get_data_cache(self):
        assert len(self._data_cache), "No data cached!"
        return self._data_cache

    @property
    def target(self):
        return self.qdesc.target

    @property
    def group_size(self):
        return self.qdesc.group_size

    @property
    def ch_axis(self):
        return self.qdesc.ch_axis


class Observer(nn.Module):
    def __init__(self, config, qdesc):
        super(Observer, self).__init__()
        self.cfg = config
        self.qdesc = qdesc
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.register_buffer("min_val", torch.tensor(float("-inf")).to(self.device))
        self.register_buffer("max_val", torch.tensor(float("inf")).to(self.device))
        self.data_cache = DataCache(qdesc)

    def calc_qparams(self):
        min_val, max_val = self.calc_minmax()
        scale, zero_point = self.calc_qparams_with_minmax(min_val, max_val)
        return scale, zero_point

    def calc_qparams_with_minmax(self, min_val, max_val):
        min_val_neg = torch.minimum(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.maximum(max_val, torch.zeros_like(max_val))
        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.float32, device=device)
        qmin, qmax = self.qdesc.qrange
        if self.is_symmetric:
            max_val_pos = torch.maximum(-min_val_neg, max_val_pos)
            scale = max_val_pos * 2 / float(qmax - qmin)
            scale = torch.maximum(scale, torch.tensor(1e-6))
        else:
            scale = (max_val_pos - min_val_neg) / float(qmax - qmin)
            scale = torch.maximum(scale, torch.tensor(1e-6))
            zero_point = torch.round(-min_val_neg / scale)
        assert len(self.data_cache) == 0, "free data cache after calc_qparams"
        return scale, zero_point

    @property
    def granularity(self):
        return self.qdesc.granularity

    @property
    def is_symmetric(self):
        return self.qdesc.is_symmetric

    @property
    def target(self):
        return self.qdesc.target

    @property
    def group_size(self):
        return self.qdesc.group_size
