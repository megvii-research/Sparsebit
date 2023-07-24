import os
import numpy as np
import torch
import torch.nn as nn
from sparsebit.quantization.common import Backend, Granularity, QuantTarget

if torch.cuda.is_available():
    from torch.utils.cpp_extension import load

    basedir = os.path.dirname(os.path.dirname(__file__))
    if not os.path.exists(os.path.join(basedir, "torch_extensions/build")):
        os.makedirs(os.path.join(basedir, "torch_extensions/build"))
    fake_quant_kernel = load(
        name="fake_quant",
        sources=[
            os.path.join(basedir, "torch_extensions/export.cc"),
            os.path.join(basedir, "torch_extensions/fake_quant_tensor.cu"),
        ],
        with_cuda=True,
        build_directory=os.path.join(basedir, "torch_extensions/build"),
        extra_cflags=["-O3"],
    )


class MySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, qdesc, backend):
        x_fq = fake_quant_factory[backend](x, scale, zero_point, qdesc)
        # my
        zp = zero_point.round()
        xq_wo_clamp = (x / scale).round() + zp
        xq = torch.clamp(xq_wo_clamp, qdesc.qmin, qdesc.qmax)
        xdq = (xq - zp) * scale
        diff = (x_fq - xdq).abs().max()
        try:
            assert diff < 1e-3, "forward large diff"
        except:
            from IPython import embed

            embed()
        ctx.save_for_backward(x, scale, zp)
        ctx.qdesc = qdesc
        return xdq

    @staticmethod
    def backward(ctx, gout):
        x, scale, zero_point = ctx.saved_tensors
        qmin, qmax = ctx.qdesc.qmin, ctx.qdesc.qmax
        xq_wo_clamp = (x / scale).round() + zero_point
        zero = gout.new_zeros(1)
        one = gout.new_ones(1)
        mask = (xq_wo_clamp >= qmin) * (xq_wo_clamp <= qmax)
        gin = torch.where(mask, gout, zero)
        gs = None
        if scale.requires_grad:
            gs = torch.where(mask, (x / scale).round() - (x / scale), zero)
            gs = (
                (xq_wo_clamp < qmin) * (qmin - zero_point)
                + (xq_wo_clamp > qmax) * (qmax - zero_point)
                + gs
            )
            gs = gout * gs
        gz = None
        if zero_point.requires_grad:
            gz = torch.where(mask, zero, one)
            gz = (
                (xq_wo_clamp < qmin) * (-scale) + (xq_wo_clamp > qmax) * (-scale)
            ) * gz
            gz = gout * gz

        return gin, gs, gz, None, None


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, qdesc, backend):
        x_fq = fake_quant_factory[backend](x, scale, zero_point, qdesc)
        ctx.save_for_backward(x, scale, zero_point)
        ctx.qdesc = qdesc
        return x_fq

    @staticmethod
    def backward(ctx, gout):
        x, scale, zero_point = ctx.saved_tensors
        qdesc = ctx.qdesc
        qmin, qmax = qdesc.qmin, qdesc.qmax
        if torch.cuda.is_available():
            if x.dtype == torch.float16:  # A workaround
                x = x.float()
            if qdesc.granularity == Granularity.CHANNELWISE:
                gx, gs, gzp = fake_quant_kernel.quant_perchannel_backward(
                    x.contiguous(),
                    scale.contiguous(),
                    zero_point.float().contiguous(),
                    gout.contiguous(),
                    qmin,
                    qmax,
                    qdesc.ch_axis,
                    0,
                )
            elif qdesc.granularity == Granularity.LAYERWISE:
                gx, gs, gzp = fake_quant_kernel.quant_pertensor_backward(
                    x.contiguous(),
                    scale,
                    zero_point.float(),
                    gout.contiguous(),
                    qmin,
                    qmax,
                    0,
                )
            else:
                raise NotImplementedError
            gs = gs if scale.requires_grad else None
            gzp = gzp if zero_point.requires_grad else None
        else:
            raise NotImplementedError(
                "We recommended that use cuda to speedup when training"
            )
            # min_fq = (qmin - zero_point) * scale
            # max_fq = (qmax - zero_point) * scale
            # zero = gout.new_zeros(1)
            # gx = torch.where((x >= min_fq) * (x <= max_fq), gout, zero)
            # if scale.requires_grad or zero_point.requires_grad:
            #    raise NotImplementedError
            # else:
            #    gs, gzp = None, None
        return gx, gs, gzp, None, None


def trt_fake_quant(x_f, scale, zero_point, qdesc):
    assert (
        x_f.device == scale.device == zero_point.device
    ), "input, scale and zero_point of quantizer must be on same device!"
    assert (
        abs(zero_point).sum() == 0
    ), "tensorrt only support symmetric quant, but zp={}".format(zero_point)
    qmin, qmax = qdesc.qrange
    if torch.cuda.is_available() and "cuda" in x_f.device.type:
        if x_f.dtype == torch.float16:  # A workaround
            x_f = x_f.float()
        if qdesc.granularity == Granularity.CHANNELWISE:
            x_dq = fake_quant_kernel.quant_perchannel_forward(
                x_f.contiguous(),
                scale.contiguous(),
                zero_point.contiguous(),
                qmin,
                qmax,
                qdesc.ch_axis,
                0,
            )
        elif qdesc.granularity == Granularity.LAYERWISE:
            x_dq = fake_quant_kernel.quant_pertensor_forward(
                x_f.contiguous(), scale, zero_point, qmin, qmax, 0
            )
        else:
            raise NotImplementedError
    else:
        x_q = torch.clamp((x_f / scale).round(), qmin, qmax)
        x_dq = x_q * scale
    return x_dq


def ort_fake_quant(x_f, scale, zero_point, qdesc):
    assert (
        x_f.device == scale.device == zero_point.device
    ), "input, scale and zero_point of quantizer must be on same device!"
    qmin, qmax = qdesc.qrange
    if torch.cuda.is_available() and "cuda" in x_f.device.type:
        if x_f.dtype == torch.float16:  # A workaround
            x_f = x_f.float()
        if qdesc.granularity == Granularity.CHANNELWISE:
            x_dq = fake_quant_kernel.quant_perchannel_forward(
                x_f.contiguous(),
                scale.contiguous(),
                zero_point.contiguous(),
                qmin,
                qmax,
                qdesc.ch_axis,
                0,
            )
        elif qdesc.granularity == Granularity.LAYERWISE:
            x_dq = fake_quant_kernel.quant_pertensor_forward(
                x_f.contiguous(), scale, zero_point, qmin, qmax, 0
            )
        elif qdesc.granularity == Granularity.GROUPWISE:
            origin_shape = x_f.shape
            grouped_shape = torch.Size([scale.numel(), -1])
            scale = scale.reshape(grouped_shape)
            zero_point = zero_point.reshape(grouped_shape)
            x_f = x_f.reshape(grouped_shape)

            x_dq = fake_quant_kernel.quant_perchannel_forward(
                x_f.contiguous(),
                scale.contiguous(),
                zero_point.contiguous(),
                qmin,
                qmax,
                0,
                0,
            )
            x_dq = x_dq.reshape(origin_shape)
        else:
            raise NotImplementedError
    else:
        if qdesc.granularity == Granularity.GROUPWISE:
            zp = zero_point.round()
            origin_shape = x_f.shape
            if qdesc.target == QuantTarget.FEATURE:
                grouped_shape = torch.Size([x_f.shape[0], scale.numel(), -1])
            else:
                grouped_shape = torch.Size([scale.numel(), -1])
            scale = scale.reshape(grouped_shape)
            zp = zp.reshape(grouped_shape)
            x_f = x_f.reshape(grouped_shape)
            x_q = torch.clamp((x_f / scale).round() + zp, qmin, qmax)
            x_dq = (x_q - zp) * scale
            x_dq = x_dq.reshape(origin_shape)

        else:
            zp = zero_point.round()
            x_q = torch.clamp((x_f / scale).round() + zp, qmin, qmax)
            x_dq = (x_q - zp) * scale
    return x_dq


fake_quant_factory = {
    Backend.VIRTUAL: ort_fake_quant,
    Backend.ONNXRUNTIME: ort_fake_quant,
    Backend.TENSORRT: trt_fake_quant,
}


def trt_dqrange(scale, zero_point, qdesc):
    assert (
        abs(zero_point).sum() == 0
    ), "tensorrt only support symmetric quant, but zp={}".format(zero_point)
    qmin, qmax = qdesc.qrange
    lower = scale * qmin
    upper = scale * qmax
    return (lower, upper)


def ort_dqrange(scale, zero_point, qdesc):
    qmin, qmax = qdesc.qrange
    lower = (qmin - zero_point) * scale
    upper = (qmax - zero_point) * scale
    return (lower, upper)


fake_qrange_factory = {
    Backend.VIRTUAL: ort_dqrange,
    Backend.ONNXRUNTIME: ort_dqrange,
    Backend.TENSORRT: trt_dqrange,
}


# torch_fake_quant仅用作模型export to onnx使用
def torch_fake_quant(x_f, scale, zero_point, qdesc):
    # lower_bound, upper_bound = qdesc.qrange
    # set [0, 255] for quint and [-128, 127] for qint because onnx only support 8 bit
    if qdesc._type.startswith("uint"):
        lower_bound, upper_bound = (0, 255)
    else:
        lower_bound, upper_bound = (-128, 127)

    if scale.numel() > 1:  # perchannel
        ch_axis = np.argmax(list(scale.shape))
        scale = scale.reshape(-1).detach().to(x_f.device)
        if torch.__version__.startswith("1.9"):  # fix bug in 1.9.x
            zero_point = zero_point.reshape(-1).long().to(x_f.device)
        else:
            zero_point = zero_point.reshape(-1).int().to(x_f.device)
        x_dq = torch.fake_quantize_per_channel_affine(
            x_f, scale, zero_point, ch_axis, lower_bound, upper_bound
        )
    elif scale.numel() == 1:  # pertensor
        scale = scale.item()
        if torch.__version__.startswith("1.9"):  # fix bug in 1.9.x
            zero_point = zero_point.long().item()
        else:
            zero_point = zero_point.int().item()
        x_dq = torch.fake_quantize_per_tensor_affine(
            x_f, scale, zero_point, lower_bound, upper_bound
        )
    else:
        raise TypeError("scale / zeropoint is not allowed to be an empty tensor")
    return x_dq
