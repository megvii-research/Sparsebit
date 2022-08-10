import torch
from yacs.config import CfgNode as CN
from sparsebit.utils.yaml_utils import _parse_config, update_config
from sparsebit.quantization.common import *

_C = CN()
_C.BACKEND = "virtual"

_C.SCHEDULE = CN()
_C.SCHEDULE.FUSE_BN = False  # use ``with torch.no_grad()`` if it's enabled
_C.SCHEDULE.DISABLE_UNNECESSARY_QUANT = True

_C.W = CN()
_C.W.QSCHEME = None  # support per-[channel/tensor]-[affine/symmetric]
_C.W.QUANTIZER = CN()
_C.W.QUANTIZER.TYPE = "uniform"
_C.W.QUANTIZER.BIT = -1
_C.W.QUANTIZER.TRAINING = CN()
_C.W.QUANTIZER.TRAINING.ENABLE = False
_C.W.QUANTIZER.TRAINING.ORDER = "together"  # support before/together/after
_C.W.QUANTIZER.TRAINING.LR = 4.0e-5
_C.W.QUANTIZER.TRAINING.ITERS = 20000
_C.W.QUANTIZER.ADAROUND = CN()
_C.W.QUANTIZER.ADAROUND.GRANULARITY = "layerwise"  # layerwise/blockwise
_C.W.QUANTIZER.ADAROUND.WARM_UP = 0.2
_C.W.QUANTIZER.ADAROUND.LAMBDA = 0.01
_C.W.QUANTIZER.ADAROUND.B_RANGE = [20, 2]
_C.W.QUANTIZER.ADAROUND.KEEP_GPU = False
_C.W.OBSERVER = CN()
_C.W.OBSERVER.TYPE = "MINMAX"  # "MINMAX"/"MSE"/"PERCENTILE"/"KL_HISTOGRAM"
_C.W.SPECIFIC = []

_C.A = CN()
_C.A.QSCHEME = None  # support per-[channel/tensor]-[affine/symmetric]
_C.A.QUANTIZER = CN()
_C.A.QUANTIZER.TYPE = "uniform"
_C.A.QUANTIZER.BIT = -1
_C.A.QUANTIZER.TRAINING = CN()
_C.A.QUANTIZER.TRAINING.ENABLE = False
_C.A.QUANTIZER.TRAINING.LR = 4.0e-5
_C.A.QUANTIZER.TRAINING.ITERS = 20000
_C.A.OBSERVER = CN()
_C.A.OBSERVER.TYPE = "MINMAX"  # "MINMAX"/"MSE"/"PERCENTILE"/"KL_HISTOGRAM"
_C.A.OBSERVER.LAYOUT = "NCHW"  # NCHW / NLC
_C.A.SPECIFIC = []


def parse_qconfig(cfg_file):
    qconfig = _parse_config(cfg_file, default_cfg=_C)
    # verify config
    verify_bits(qconfig)
    verify_backend(qconfig)
    return qconfig


def verify_bits(qconfig):
    assert (
        qconfig.W.QUANTIZER.BIT >= 0
    ), "bitwidth of weight shoud be a non-negative number"
    assert (
        qconfig.A.QUANTIZER.BIT >= 0
    ), "bitwidth of activation shoud be a non-negative number"
    # TODO verify the bit in specific


def verify_backend(qconfig):
    backend = get_backend(qconfig.BACKEND)
    w_qscheme = get_qscheme(qconfig.W.QSCHEME)
    a_qscheme = get_qscheme(qconfig.A.QSCHEME)
    if backend in [Backend.ONNXRUNTIME, Backend.TENSORRT]:
        wbit = qconfig.W.QUANTIZER.BIT
        abit = qconfig.A.QUANTIZER.BIT
        assert (
            wbit == 8 and abit == 8
        ), "onnxruntime/tensorrt only support bit=8, if <8bit, we recommend use 'virtual' as backbend"
    if backend == Backend.TENSORRT:
        assert (
            w_qscheme == torch.per_channel_symmetric
        ), "the qshema of weight should be specified as per-channel-symmetric in tensorrt"
        assert (
            a_qscheme == torch.per_tensor_symmetric
        ), "the qshema of activation should be specified as per-tensor-symmetric in tensorrt"
