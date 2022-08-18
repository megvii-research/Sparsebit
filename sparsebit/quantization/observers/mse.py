import torch
import math
from sparsebit.quantization.observers import Observer as BaseObserver
from sparsebit.quantization.observers import register_observer
from sparsebit.quantization.quantizers.quant_tensor import STE
from sparsebit.quantization.common import Backend


@register_observer
class Observer(BaseObserver):
    TYPE = "mse"

    def __init__(self, config, qdesc):
        super(Observer, self).__init__(config, qdesc)
        self.alpha = config.OBSERVER.PERCENTILE.ALPHA

    def _mse_loss(self, pred, tgt):
        """
        loss function measured in L_p Norm
        """
        return (pred - tgt).abs().pow(2).mean()

    def calc_minmax(self):
        data_c_first = self.get_calibration_data(c_first=True)
        if self.is_perchannel:
            max_val = data_c_first.max(axis=1).values
            min_val = data_c_first.min(axis=1).values
        else:
            min_val, max_val = data_c_first.min(), data_c_first.max()
        self.min_val = min_val.to(self.device)
        self.max_val = max_val.to(self.device)
        return self.min_val, self.max_val

    def calc_qparams(self):
        qmin, qmax = self.qdesc.qrange
        x_f = torch.cat(self._data_cache, axis=0)
        min_val, max_val = self.calc_minmax()
        device = min_val.device
        x_f = x_f.to(device)
        best_scale, best_zero_point = None, None
        loss_min = 1e10
        for i in range(80):
            cur_min_val = min_val * (1.0 - (i * 0.01))
            cur_max_val = max_val * (1.0 - (i * 0.01))
            scale, zero_point = self.calc_qparams_with_minmax(cur_min_val, cur_max_val)
            x_dq = STE.apply(x_f, scale, zero_point, self.qdesc, Backend.VIRTUAL)
            loss = self._mse_loss(x_f, x_dq)
            if loss < loss_min:
                loss_min = loss
                best_scale = scale
                best_zero_point = zero_point
        self.reset_data_cache()
        assert len(self._data_cache) == 0, "free data cache after calc_qparams"
        return best_scale, best_zero_point
