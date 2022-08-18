import torch
import math
from sparsebit.quantization.observers import Observer as BaseObserver
from sparsebit.quantization.observers import register_observer
from sparsebit.quantization.quantizers.quant_tensor import STE


def mse_loss(pred, tgt, is_perchannel):
    if is_perchannel:
        return ((pred - tgt) ** 2).mean(-1)
    else:
        return ((pred - tgt) ** 2).mean()


@register_observer
class Observer(BaseObserver):
    TYPE = "aciq"

    def __init__(self, config, qdesc):
        super(Observer, self).__init__(config, qdesc)

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

    def calc_laplace_minmax(self, data, is_half_range):
        if self.is_perchannel:
            b = torch.mean(torch.abs(data - data.mean(1).unsqueeze(1)), dim=1)
        else:
            b = torch.mean(torch.abs(data - data.mean()))
        if (
            self.qdesc.scheme in [torch.per_channel_affine, torch.per_tensor_affine]
            and is_half_range
        ):
            max_val = self.alpha_laplace_positive[self.qdesc.bit] * b
            min_val = torch.zeros(max_val.shape)
        else:
            max_val = self.alpha_laplace[self.qdesc.bit] * b
            min_val = -max_val
        return min_val, max_val

    def calc_gaus_minmax(self, data, batch_size, is_half_range):
        if self.is_perchannel:
            max_val = data.max(axis=1).values
            min_val = data.min(axis=1).values
        else:
            max_val = data.max()
            min_val = data.min()
        num_elements_per_batch = data.numel()
        if self.qdesc.ch_axis > 0:
            num_elements_per_batch /= batch_size
        std = ((max_val - min_val) * self.gaus_const) / (
            (2 * math.log(num_elements_per_batch)) ** 0.5
        )
        if (
            self.qdesc.scheme in [torch.per_channel_affine, torch.per_tensor_affine]
            and is_half_range
        ):
            max_val = self.alpha_gaus_positive[self.qdesc.bit] * std
            min_val = torch.zeros(max_val.shape)
        else:
            max_val = self.alpha_gaus[self.qdesc.bit] * std
            min_val = -max_val
        return min_val, max_val

    def calc_naive_minmax(self, data):
        if self.is_perchannel:
            max_val = data.max(axis=1).values
            min_val = data.min(axis=1).values
        else:
            max_val = data.max()
            min_val = data.min()
        return min_val, max_val

    def calc_minmax(self):
        batch_size = (
            torch.cat(self._data_cache, axis=0).shape[0]
            if self.qdesc.ch_axis > 0
            else 1
        )
        data = self.get_calibration_data(c_first=True)
        is_half_range = data.min() >= 0

        laplace_min_val, laplace_max_val = self.calc_laplace_minmax(data, is_half_range)
        scale_laplace, zero_point_laplace = self.calc_qparams_with_minmax(
            laplace_min_val, laplace_max_val
        )
        mse_laplace = mse_loss(
            STE.apply(
                data, scale_laplace, zero_point_laplace, self.qdesc, self.backend
            ),
            data,
            self.is_perchannel,
        )

        gaus_min_val, gaus_max_val = self.calc_gaus_minmax(
            data, batch_size, is_half_range
        )
        scale_gaus, zero_point_gaus = self.calc_qparams_with_minmax(
            gaus_min_val, gaus_max_val
        )

        mse_gaus = mse_loss(
            STE.apply(data, scale_gaus, zero_point_gaus, self.qdesc, self.backend),
            data,
            self.is_perchannel,
        )

        naive_min_val, naive_max_val = self.calc_naive_minmax(data)
        scale_minmax, zero_point_minmax = self.calc_qparams_with_minmax(
            naive_min_val, naive_max_val
        )
        mse_minmax = mse_loss(
            STE.apply(data, scale_minmax, zero_point_minmax, self.qdesc, self.backend),
            data,
            self.is_perchannel,
        )

        mse_gaus_laplace = torch.where(mse_gaus < mse_laplace, mse_gaus, mse_laplace)
        self.min_val = torch.where(
            mse_gaus < mse_laplace, gaus_min_val, laplace_min_val
        )
        self.min_val = torch.where(
            mse_minmax < mse_gaus_laplace, naive_min_val, self.min_val
        ).to(self.device)
        self.max_val = torch.where(
            mse_gaus < mse_laplace, gaus_max_val, laplace_max_val
        )
        self.max_val = torch.where(
            mse_minmax < mse_gaus_laplace, naive_max_val, self.max_val
        ).to(self.device)

        return self.min_val, self.max_val
