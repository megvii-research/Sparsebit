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
        )
        if groupsize == -1:
            q8weight = cuda_kernel.unpack(qweight, (zeros / scales).round().char(), True)
            q_input, inv_scale = cuda_kernel.quant_pertoken(input)
            cuda_kernel.int8gemm(
                q_input.reshape(-1, q_input.shape[-1]), q8weight, y, 1.0, 0.0
            )
            y = y.reshape(x_shape[0:2] + [-1])
            y *= inv_scale.unsqueeze(-1)
            y *= scales[:, 0]
        else:
            raise NotImplementedError
        ctx.scales = scales
        ctx.zeros = zeros
        ctx.qweight = qweight
        ctx.in_shapes = list(input.shape)
        ctx.backward_ic_scales = backward_ic_scales
        ctx.backward_ic_zeros = backward_ic_zeros
        return y

    @staticmethod
    def backward(ctx, grad_y):
        ic_scales = ctx.backward_ic_scales
        ic_zeros = ctx.backward_ic_zeros
        ic_shapes = [1] * (len(ctx.in_shapes) - 1) + [ctx.in_shapes[-1]]
        assert ic_scales is not None and ic_zeros is not None
        scales = ctx.scales
        zeros = ctx.zeros
        qweight = ctx.qweight
        grad_x = torch.zeros(ctx.in_shapes, dtype=grad_y.dtype, device=grad_y.device)
        grad_y8, inv_grad_scales = cuda_kernel.quant_pertoken(grad_y)
        qweight8_ic_t = cuda_kernel.unpack_backward(
            qweight, scales, zeros, ic_scales, ic_zeros, False
        )
        cuda_kernel.int8gemm(
            grad_y8.view(-1, grad_y8.size(-1)),
            qweight8_ic_t,
            grad_x.view(-1, grad_x.size(-1)),
            1.0,
            0.0,
        )
        grad_x *= ic_scales.view(ic_shapes)
        grad_x -= torch.sum(grad_y8, dim=-1, keepdim=True) * ic_zeros.view(ic_shapes)
        grad_x *= inv_grad_scales.unsqueeze(-1)

        return grad_x, None, None, None, None, None, None, None

