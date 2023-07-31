import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as ddp

from sparsebit.quantization.quantizers import Quantizer as BaseQuantizer
from sparsebit.quantization.quantizers import register_quantizer
from sparsebit.quantization.common import Granularity
from .lsq import gs_scaling, STE


@register_quantizer
class Quantizer(BaseQuantizer):
    TYPE = "LSQ+"

    def __init__(self, config):
        super(Quantizer, self).__init__(config)
        self.init_params = False

    def calc_qparams(self):
        if self.fake_fused:
            return self.scale, self.zero_point
        if not self.init_params:
            if self.granularity == Granularity.CHANNELWISE:
                x_oc = self.observer.data_cache.get_data_for_calibration(
                    Granularity.CHANNELWISE
                )
                assert (
                    self.is_symmetric
                ), "LSQ+ only support per-channel-symmetric quant for weight"
                mean, std = x_oc.mean(axis=1), x_oc.std(axis=1)
                scale = (
                    2
                    * torch.maximum((mean - 3 * std).abs(), (mean + 3 * std).abs())
                    / (self.qdesc.qmax - self.qdesc.qmin)
                )
                self.scale = nn.Parameter(
                    self._broadcast_qparams(scale.to(self.device))
                )
                self.zero_point = self._broadcast_qparams(torch.zeros_like(self.scale))
            elif granularity == Granularity.LAYERWISE:
                assert (
                    not self.is_symmetric
                ), "LSQ+ only support per-tensor-affine quant for activation"
                scale, zero_point = self.observer.calc_qparams()
                self.scale = nn.Parameter(
                    self._broadcast_qparams(scale.to(self.device))
                )
                zero_point = zero_point.clamp(self.qdesc.qmin, self.qdesc.qmax)
                self.zero_point = nn.Parameter(
                    self._broadcast_qparams(zero_point.to(self.device))
                )
            else:
                raise NotImplementedError
            self.init_params = True
        return self.scale, self.zero_point

    def _qparams_preprocess(self, x):
        if self.export_onnx:
            return torch.tensor(
                self.scale.abs().detach().cpu().numpy(), device=self.device
            ), torch.tensor(
                torch.clamp(self.zero_point, self.qdesc.qmin, self.qdesc.qmax)
                .detach()
                .cpu()
                .numpy(),
                device=self.device,
            )
        scale = self.scale.abs()
        zero_point = torch.clamp(self.zero_point, self.qdesc.qmin, self.qdesc.qmax)
        return scale, zero_point

    def _forward(self, x, scale, zero_point):
        if self.granularity == Granularity.CHANNELWISE:
            num_perchannel = x.numel() / x.shape[self.qdesc.ch_axis]
            gs_ratio = 1.0 / math.sqrt(num_perchannel * self.qdesc.qmax)
        elif self.granularity == Granularity.LAYERWISE:
            gs_ratio = 1.0 / math.sqrt(x.numel() * self.qdesc.qmax)
        else:
            raise NotImplementedError
        scale = gs_scaling.apply(scale, gs_ratio)
        if zero_point.requires_grad:
            zero_point = gs_scaling.apply(zero_point, gs_ratio)
        x_dq = STE.apply(x, scale, zero_point, self.qdesc, self.backend)
        return x_dq
