import torch
from sparsebit.quantization.common import get_qscheme


class QuantDescriptor:
    def __init__(self, cfg):
        self._cfg = cfg
        self._scheme = get_qscheme(cfg.QSCHEME)
        self._bit = cfg.QUANTIZER.BIT
        self._qmin, self._qmax, self._type = self.calc_qmin_qmax(
            self._bit, self._scheme
        )
        self._ch_axis = self._set_channel_axis()
        self.is_perchannel = (
            self._scheme == torch.per_channel_symmetric
            or self._scheme == torch.per_channel_affine
        )
        self.is_symmetric = (
            self._scheme == torch.per_channel_symmetric
            or self._scheme == torch.per_tensor_symmetric
        )

    def calc_qmin_qmax(self, bit, scheme):
        if scheme in [torch.per_channel_symmetric, torch.per_tensor_symmetric]:
            qmin = -(2 ** (bit - 1))
            qmax = 2 ** (bit - 1) - 1
            _type = "int{}".format(bit)
        elif scheme in [torch.per_channel_affine, torch.per_tensor_affine]:
            qmin = 0
            qmax = 2 ** bit - 1
            _type = "uint{}".format(bit)
        return qmin, qmax, _type

    def _set_channel_axis(self):
        if hasattr(self._cfg.OBSERVER, "LAYOUT"):  # activation
            layout = self._cfg.OBSERVER.LAYOUT
            if layout == "NCHW":  # for cnn
                ch_axis = 1
            elif layout == "NLC":  # for transformer
                ch_axis = 2
            else:
                raise NotImplementedError
        else:  # weight
            ch_axis = 0
        return ch_axis

    def set_bit(self, bit):
        self._bit = bit
        self._qmin, self._qmax, self._type = self.calc_qmin_qmax(bit, self._scheme)

    def reset_scheme(self, is_symmetric = None, is_perchannel = None):
        assert is_symmetric is not None or is_perchannel is not None, "A target must be set for reseting scheme!"
        if (is_symmetric, is_perchannel) in [(self.is_symmetric, self.is_perchannel),(self.is_symmetric, None),(None, self.is_perchannel)]:
            return
        if is_symmetric is None:
            is_symmetric = self.is_symmetric
        if is_perchannel is None:
            is_perchannel = self.is_perchannel
        self._scheme = {
            (True, True): torch.per_channel_symmetric,
            (True, False): torch.per_tensor_symmetric,
            (False, True): torch.per_channel_affine,
            (False, False): torch.per_tensor_affine,
        }[(is_symmetric, is_perchannel)]
        self._qmin, self._qmax, self._type = self.calc_qmin_qmax(
            self._bit, self._scheme
        )
        self._ch_axis = self._set_channel_axis()

    @property
    def scheme(self):
        return self._scheme

    @property
    def bit(self):
        return self._bit

    @property
    def qmin(self):
        return self._qmin

    @property
    def qmax(self):
        return self._qmax

    @property
    def qrange(self):
        return (self._qmin, self._qmax)

    @property
    def ch_axis(self):
        return self._ch_axis

    def __repr__(self):
        return self._type + "\t qmin: {}  qmax: {}, qscheme: {}".format(
            self.qmin, self.qmax, self.scheme
        )
