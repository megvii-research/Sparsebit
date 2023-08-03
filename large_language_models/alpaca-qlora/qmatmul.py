import torch
from load_cuda_kernel import cuda_kernel


class Quant4Matmul(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        qweight,
        scales,
        zeros,
        bias,
        groupsize=-1,
        backward_ic_scales=None,
        backward_ic_zeros=None,
    ):
        x_shape = list(input.shape)
        y = (
            bias.to(input.dtype)[(None,) * (len(x_shape) - 1)]
            .repeat(x_shape[:-1] + [1])
            .reshape(-1, qweight.shape[-1])
            .contiguous()
        ).reshape(x_shape[0:2] + [-1])
        if groupsize == -1:
            q8weight = cuda_kernel.unpack(
                qweight, (zeros / scales).round().char(), True
            )
            input, inv_scale = cuda_kernel.quant_pertoken(input)
            cuda_kernel.int8gemm(
                input.reshape(-1, input.shape[-1]), q8weight, y, 1.0, 0.0
            )
            y = y.reshape(x_shape[0:2] + [-1])
            y *= inv_scale.unsqueeze(-1)
            y *= scales[:, 0]
        else:
            raise NotImplementedError
        ctx.save_for_backward(
            qweight, scales, zeros, backward_ic_scales, backward_ic_zeros
        )
        ctx.in_shapes = list(input.shape)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        qweight, scales, zeros, ic_scales, ic_zeros = ctx.saved_tensors
        grad_x = torch.zeros(ctx.in_shapes, dtype=grad_y.dtype, device=grad_y.device)
        fp16_weight = cuda_kernel.unpack(
            qweight, (zeros / scales).round().char(), True
        ).to(grad_y.dtype)
        fp16_weight = fp16_weight * scales
        grad_x = torch.matmul(grad_y, fp16_weight)
        return grad_x, None, None, None, None, None, None, None
