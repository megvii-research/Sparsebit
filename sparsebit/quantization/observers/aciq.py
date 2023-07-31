import torch
import math
from sparsebit.quantization.observers import Observer as BaseObserver
from sparsebit.quantization.observers import register_observer
from sparsebit.quantization.quantizers.quant_tensor import STE
from sparsebit.quantization.common import Granularity, QuantTarget


@register_observer
class Observer(BaseObserver):
    TYPE = "aciq"

    def __init__(self, config, qdesc):
        super(Observer, self).__init__(config, qdesc)

        self.distribution = config.OBSERVER.ACIQ.DISTRIBUTION.lower()
        assert self.distribution in [
            "gaus",
            "laplace",
        ], "ACIQ observer only support 'gaus' and 'laplace' mode!"
        self.alpha_gaus_positive = {
            1: 1.71,
            2: 2.15,
            3: 2.55,
            4: 2.93,
            5: 3.28,
            6: 3.61,
            7: 3.92,
            8: 4.2,
        }
        self.alpha_gaus = {
            1: 1.24,
            2: 1.71,
            3: 2.15,
            4: 2.55,
            5: 2.93,
            6: 3.28,
            7: 3.61,
            8: 3.92,
        }
        self.alpha_laplace = {
            0: 1.05,
            1: 1.86,
            2: 2.83,
            3: 3.89,
            4: 5.03,
            5: 6.2,
            6: 7.41,
            7: 8.64,
            8: 9.89,
        }
        self.alpha_laplace_positive = {
            0: 1.86,
            1: 2.83,
            2: 3.89,
            3: 5.02,
            4: 6.2,
            5: 7.41,
            6: 8.64,
            7: 9.89,
            8: 11.16,
        }
        self.gaus_const = (0.5 * 0.35) * (1 + (math.pi * math.log(4)) ** 0.5)

    def calc_laplace_minmax(self):
        data = self.data_cache.get_data_for_calibration(self.granularity)
        if self.granularity in [Granularity.CHANNELWISE, Granularity.GROUPWISE]:
            b = torch.mean(torch.abs(data - data.mean(1).unsqueeze(1)), dim=1)
        elif self.granularity == Granularity.LAYERWISE:
            b = torch.mean(torch.abs(data - data.mean()))
        else:
            raise NotImplementedError
        self.data_cache.reset()
        is_half_range = data.min() >= 0
        if (
            self.qdesc.scheme
            in [torch.per_channel_affine, torch.per_tensor_affine, "per-group-affine"]
            and is_half_range
        ):
            max_val = self.alpha_laplace_positive[self.qdesc.bit] * b
            min_val = torch.zeros(max_val.shape)
        else:
            max_val = self.alpha_laplace[self.qdesc.bit] * b
            min_val = -max_val
        return min_val, max_val

    def calc_gaus_minmax(self):
        if self.qdesc.target == QuantTarget.FEATURE:
            batch_size = self.data_cache.get_batch_size()
        data = self.data_cache.get_data_for_calibration(self.granularity)
        if self.granularity in [Granularity.CHANNELWISE, Granularity.GROUPWISE]:
            max_val = data.max(axis=1).values
            min_val = data.min(axis=1).values
        elif Granularity.LAYERWISE:
            max_val = data.max()
            min_val = data.min()
        else:
            raise NotImplementedError
        self.data_cache.reset()
        is_half_range = data.min() >= 0
        num_elements = data[0].numel()
        if self.qdesc.target == QuantTarget.FEATURE:
            num_elements /= batch_size
        std = ((max_val - min_val) * self.gaus_const) / (
            (2 * math.log(num_elements)) ** 0.5
        )
        if (
            self.qdesc.scheme
            in [torch.per_channel_affine, torch.per_tensor_affine, "per-group-affine"]
            and is_half_range
        ):
            max_val = self.alpha_gaus_positive[self.qdesc.bit] * std
            min_val = torch.zeros(max_val.shape)
        else:
            max_val = self.alpha_gaus[self.qdesc.bit] * std
            min_val = -max_val
        return min_val, max_val

    def calc_minmax(self):
        if self.distribution == "laplace":
            min_val, max_val = self.calc_laplace_minmax()
        else:
            min_val, max_val = self.calc_gaus_minmax()
        self.min_val = min_val.to(self.device)
        self.max_val = max_val.to(self.device)

        return self.min_val, self.max_val
