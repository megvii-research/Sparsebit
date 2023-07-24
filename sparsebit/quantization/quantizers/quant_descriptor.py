import torch
from sparsebit.quantization.common import get_qscheme, Granularity


class QuantDescriptor:
    def __init__(self, cfg):
        self._cfg = cfg
        self._target = cfg.TARGET[0]
        self._scheme = get_qscheme(cfg.QSCHEME)
        self._bit = cfg.QUANTIZER.BIT
        self._group_size = cfg.QUANTIZER.GROUP_SIZE
        if self._group_size != -1:
            assert cfg.QSCHEME in ["per-group-symmetric", "per-group-affine"]
        self._qmin, self._qmax, self._type = self.calc_qmin_qmax(
            self._bit, self._scheme
        )
        self._ch_axis = self._set_channel_axis()
        self._bs_axis = self._set_batchsize_axis()
        self.granularity = {
            torch.per_channel_symmetric: Granularity.CHANNELWISE,
            torch.per_channel_affine: Granularity.CHANNELWISE,
            torch.per_tensor_symmetric: Granularity.LAYERWISE,
            torch.per_tensor_affine: Granularity.LAYERWISE,
            "per-group-symmetric": Granularity.GROUPWISE,
            "per-group-affine": Granularity.GROUPWISE,
        }[self._scheme]
        self.is_symmetric = (
            self._scheme == torch.per_channel_symmetric
            or self._scheme == torch.per_tensor_symmetric
            or self._scheme == "per-group-symmetric"
        )

    def calc_qmin_qmax(self, bit, scheme):
        if scheme in [
            torch.per_channel_symmetric,
            torch.per_tensor_symmetric,
            "per-group-symmetric",
        ]:
            qmin = -(2 ** (bit - 1))
            qmax = 2 ** (bit - 1) - 1
            _type = "int{}".format(bit)
        elif scheme in [
            torch.per_channel_affine,
            torch.per_tensor_affine,
            "per-group-affine",
        ]:
            qmin = 0
            qmax = 2**bit - 1
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

    def _set_batchsize_axis(self):
        if hasattr(self._cfg.OBSERVER, "LAYOUT"):  # activation
            layout = self._cfg.OBSERVER.LAYOUT
            if layout in ["NCHW", "NLC"]:  # for cnn
                bs_axis = 0
            else:
                raise NotImplementedError
        else:  # weight
            bs_axis = None
        return bs_axis

    def set_bit(self, bit):
        self._bit = bit
        self._qmin, self._qmax, self._type = self.calc_qmin_qmax(bit, self._scheme)

    def set_group_size(self, group_size):
        self._group_size = group_size

    def set_symmetric(self, is_symmetric: bool):
        self.is_symmetric = is_symmetric
        self._scheme = {
            (Granularity.CHANNELWISE, True): torch.per_channel_symmetric,
            (Granularity.CHANNELWISE, False): torch.per_channel_affine,
            (Granularity.LAYERWISE, True): torch.per_tensor_symmetric,
            (Granularity.LAYERWISE, False): torch.per_tensor_affine,
            (Granularity.GROUPWISE, True): "per-group-symmetric",
            (Granularity.GROUPWISE, False): "per-group-affine",
        }[(self.granularity, self.is_symmetric)]
        self._qmin, self._qmax, self._type = self.calc_qmin_qmax(
            self._bit, self._scheme
        )

    @property
    def target(self):
        return self._target

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

    @property
    def bs_axis(self):
        return self._bs_axis

    @property
    def group_size(self):
        return self._group_size

    def __repr__(self):
        return self._type + "\t qmin: {}  qmax: {}, qscheme: {}, group_size: {}".format(
            self.qmin, self.qmax, self.scheme, self.group_size
        )
