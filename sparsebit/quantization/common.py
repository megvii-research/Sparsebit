import torch
from enum import Enum


class Granularity(Enum):
    LAYERWISE = 0
    CHANNELWISE = 1


class QuantTarget(Enum):
    WEIGHT = 0
    FEATURE = 1


class Backend(Enum):
    VIRTUAL = 0
    ONNXRUNTIME = 1
    TENSORRT = 2


def get_backend(backend):
    target_backend = [
        "onnxruntime",
        "tensorrt",
        "virtual",  # 虚拟平台, 无法部署, 用于研究<8bit使用
    ]
    if backend == "virtual":
        return Backend.VIRTUAL
    if backend == "onnxruntime":
        return Backend.ONNXRUNTIME
    if backend == "tensorrt":
        return Backend.TENSORRT
    raise TypeError(
        "only support backend in {}, not {}".format(target_backend, backend)
    )


def get_qscheme(qscheme):
    if qscheme == "per-tensor-symmetric":
        return torch.per_tensor_symmetric
    if qscheme == "per-tensor-affine":
        return torch.per_tensor_affine
    if qscheme == "per-channel-symmetric":
        return torch.per_channel_symmetric
    if qscheme == "per-channel-affine":
        return torch.per_channel_affine
    raise TypeError(
        "only support a qscheme equals to per-[tensor/channel]-[affine/symmetric] , not {}".format(
            qscheme
        )
    )
