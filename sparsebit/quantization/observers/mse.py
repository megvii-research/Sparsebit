import torch
import math
from .utils import mse_loss
from sparsebit.quantization.observers import Observer as BaseObserver
from sparsebit.quantization.observers import register_observer
from sparsebit.quantization.quantizers.quant_tensor import STE
from sparsebit.quantization.common import Backend, Granularity


@register_observer
class Observer(BaseObserver):
    TYPE = "mse"

    def __init__(self, config, qdesc):
        super(Observer, self).__init__(config, qdesc)
        self.alpha = config.OBSERVER.PERCENTILE.ALPHA

    def calc_minmax(self, data_c_first):
        if self.granularity in [Granularity.CHANNELWISE, Granularity.GROUPWISE]:
            max_val = data_c_first.max(axis=1).values
            min_val = data_c_first.min(axis=1).values
        elif self.granularity == Granularity.LAYERWISE:
            min_val, max_val = data_c_first.min(), data_c_first.max()
        else:
            raise NotImplementedError
        self.min_val = min_val.to(self.device)
        self.max_val = max_val.to(self.device)
        return self.min_val, self.max_val

    def calc_qparams(self):
        data_c_first = self.data_cache.get_data_for_calibration(self.granularity)
        self.data_cache.reset()
        min_val, max_val = self.calc_minmax(data_c_first)
        x_f = data_c_first.to(self.device)
        if self.granularity in [Granularity.CHANNELWISE, Granularity.GROUPWISE]:
            best_scale = torch.tensor(
                [1.0 for _ in range(data_c_first.shape[0])], device=self.device
            )
            best_zero_point = torch.tensor(
                [0.0 for _ in range(data_c_first.shape[0])], device=self.device
            )
            loss_min = torch.tensor(
                [1e10 for _ in range(data_c_first.shape[0])], device=self.device
            )
        elif self.granularity == Granularity.LAYERWISE:
            best_scale, best_zero_point = None, None
            loss_min = 1e10
        else:
            raise NotImplementedError
        for i in range(80):
            cur_min_val = min_val * (1.0 - (i * 0.01))
            cur_max_val = max_val * (1.0 - (i * 0.01))
            scale, zero_point = self.calc_qparams_with_minmax(cur_min_val, cur_max_val)
            x_dq = STE.apply(x_f, scale, zero_point, self.qdesc, Backend.VIRTUAL)
            loss = mse_loss(x_f, x_dq, self.granularity)
            if self.granularity in [Granularity.CHANNELWISE, Granularity.GROUPWISE]:
                best_scale[loss < loss_min] = scale[loss < loss_min]
                best_zero_point[loss < loss_min] = zero_point[loss < loss_min]
                loss_min[loss < loss_min] = loss[loss < loss_min]
            elif self.granularity == Granularity.LAYERWISE:
                if loss < loss_min:
                    loss_min = loss
                    best_scale = scale
                    best_zero_point = zero_point
            else:
                raise NotImplementedError
        assert len(self.data_cache) == 0, "free data cache after calc_qparams"
        return best_scale, best_zero_point
