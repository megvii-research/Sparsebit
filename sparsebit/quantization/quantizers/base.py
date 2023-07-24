import torch
from torch import nn
import abc
from sparsebit.quantization.observers import build_observer
import warnings
from .quant_tensor import torch_fake_quant
from .quant_descriptor import QuantDescriptor


class Quantizer(nn.Module, abc.ABC):
    def __init__(self, config):
        super(Quantizer, self).__init__()
        self.cfg = config
        self.qdesc = QuantDescriptor(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.register_buffer(
            "scale", torch.tensor([1.0], dtype=torch.float).to(self.device)
        )
        self.register_buffer(
            "zero_point", torch.tensor([0.0], dtype=torch.float).to(self.device)
        )
        self.observer = build_observer(config, self.qdesc)
        self.use_quant = False
        self.export_onnx = False
        self.fake_fused = False
        if self.cfg.QUANTIZER.DISABLE:
            self.set_fake_fused()
        if self.qdesc.bit == 0:
            warnings.warn(
                "used bit==0 to disable quantizer is deprecated, please use a flag: QUANTIZER.DISABLE"
            )

    def calc_qparams(self):
        if self.fake_fused:
            return self.scale, self.zero_point
        scale, zero_point = self.observer.calc_qparams()
        self.scale = self._broadcast_qparams(scale)
        self.zero_point = self._broadcast_qparams(zero_point)
        return self.scale, self.zero_point

    def calc_qparams_with_minmax(self, min_val, max_val):
        if self.fake_fused:
            return self.scale, self.zero_point
        scale, zero_point = self.observer.calc_qparams_with_minmax(min_val, max_val)
        self.scale = self._broadcast_qparams(scale)
        self.zero_point = self._broadcast_qparams(zero_point)
        return self.scale, self.zero_point

    def _forward(self, x, scale, zero_point):
        pass

    def _qparams_preprocess(self, x):
        return self.scale, self.zero_point

    def forward(self, x):
        if self.is_enable:
            scale, zero_point = self._qparams_preprocess(x)
            if self.export_onnx:
                x_dq = torch_fake_quant(x, scale, zero_point, self.qdesc)
            else:
                x_dq = self._forward(x, scale, zero_point)
        else:
            x_dq = x
        return x_dq

    def update_observer(self, x):
        self.dims = len(x.shape)
        self.observer.data_cache.update(x.detach())

    def set_backend(self, backend):
        self.backend = backend
        self.observer.backend = backend

    def set_fake_fused(self):
        self.fake_fused = True
        if isinstance(self.scale, nn.Parameter):
            self.scale.requires_grad_(False)
            self.zero_point.requires_grad_(False)
        else:
            self.scale = torch.tensor([1.0], dtype=torch.float).to(self.device)
            self.zero_point = torch.tensor([0.0], dtype=torch.float).to(self.device)

    def enable_quant(self):
        self.use_quant = True

    def disable_quant(self):
        self.use_quant = False

    def enable_export_onnx(self):
        self.export_onnx = True
        # round zero point for onnx export
        self.zero_point = self.zero_point.round()

    def disable_export_onnx(self):
        self.export_onnx = False

    def _broadcast_qparams(self, params):
        dst_shape = [1] * self.dims
        dst_shape[self.qdesc.ch_axis] = -1
        return params.reshape(dst_shape)

    def set_bit(self, bit):
        self.qdesc.set_bit(bit)

    @property
    def is_enable(self):
        return self.use_quant and (not self.fake_fused)

    @property
    def bit(self):
        return self.qdesc.bit

    @property
    def ch_axis(self):
        return self.observer.ch_axis

    @property
    def granularity(self):
        return self.qdesc.granularity

    @property
    def is_symmetric(self):
        return self.qdesc.is_symmetric

    def __repr__(self):
        info = "{}, {}, observer={},".format(self.TYPE, self.qdesc, self.observer.TYPE)
        if self.qdesc.scheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
            info += " scale={:.4f}, zp={:.4f}".format(
                self.scale.item(), self.zero_point.item()
            )
        elif self.qdesc.scheme in [
            torch.per_channel_affine,
            torch.per_channel_symmetric,
        ]:
            info += " scale=[{:.4f}, {:.4f}], zp=[{}, {}]".format(
                self.scale.min(),
                self.scale.max(),
                self.zero_point.min(),
                self.zero_point.max(),
            )
        else:
            raise NotImplementedError
        return info
