"""
quantize / dequantize is used in onnx export, with lower bits supported
in sparsebit.quantization.quantizers.quant_tensor.py
"""
import torch
from torch.onnx.symbolic_helper import parse_args
import torch.onnx.symbolic_helper as sym_help


def analyze_min_max(L, R):
    valid_symmetric_ranges = {
        (-(2 ** (i - 1)), 2 ** (i - 1) - 1): i for i in range(2, 9)
    }
    valid_asymmetric_ranges = {(0, 2**i - 1): i for i in range(2, 9)}
    valid_asymmetric_ranges[(0, 1)] = 1
    if (L, R) in valid_symmetric_ranges:
        return valid_symmetric_ranges[(L, R)], True
    elif (L, R) in valid_asymmetric_ranges:
        return valid_asymmetric_ranges[(L, R)], False

    # no valid range in <=8bit types
    return None, None


@parse_args("v", "v", "v", "i", "i", "i", "b")
def onnx_quantize(
    g, inputs, scale, zero_point, axis, quant_min=-128, quant_max=127, extra_info=False
):
    bit, is_symmetric = analyze_min_max(quant_min, quant_max)
    assert (
        bit is not None and is_symmetric is not None
    ), "the range ({}, {}) does not identify a valid data_type with bits<=8".format(
        quant_min, quant_max
    )

    if isinstance(scale, float):
        scale = torch.tensor(scale)
        scale = scale.to(torch.float32)
    if isinstance(zero_point, int):
        zero_point = torch.tensor(zero_point)
        if is_symmetric:
            zero_point = zero_point.to(torch.int8)
        else:
            zero_point = zero_point.to(torch.uint8)

    kwargs = {"axis_i": axis}
    quant_op_name = "QuantizeLinear"
    dequant_op_name = "DequantizeLinear"
    if extra_info:
        kwargs["dtype_s"] = "{}int{}".format("s" if is_symmetric else "u", str(bit))
        kwargs["bits_i"] = bit
        # change the operator domain, to avoid onnx.checker.check_model failure
        quant_op_name = "Sparsebit::{}".format(quant_op_name)
        dequant_op_name = "Sparsebit::{}".format(dequant_op_name)
    quant_op = g.op(quant_op_name, inputs, scale, zero_point, **kwargs)
    dequant_op = g.op(dequant_op_name, quant_op, scale, zero_point, **kwargs)
    return dequant_op


class QuantizeFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs,
        scale,
        zero_point,
        axis,
        quant_min=-128,
        quant_max=127,
        per_channel=True,
        extra_info=False,
    ):
        if per_channel:
            return torch.fake_quantize_per_channel_affine(
                inputs, scale, zero_point, axis, quant_min, quant_max
            )
        else:
            return torch.fake_quantize_per_tensor_affine(
                inputs, scale, zero_point, quant_min, quant_max
            )

    @staticmethod
    def backward(ctx, grad):
        return (None,) * 8

    @staticmethod
    def symbolic(
        g: torch.Graph,
        inputs: torch.Value,
        scale: torch.Value,
        zero_point: torch.Value,
        axis: int,
        quant_min: int = -128,
        quant_max: int = 127,
        per_channel: bool = True,
        extra_info: bool = False,
    ):
        return onnx_quantize(
            g, inputs, scale, zero_point, axis, quant_min, quant_max, extra_info
        )
