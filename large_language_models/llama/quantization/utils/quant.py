import numpy as np
import torch
import torch.nn as nn

from .load_cuda_kernel import cuda_kernel


def quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


def ceiling_div(x, y):
    return (x + y - 1) // y


class Quantizer(nn.Module):
    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bit,
        perchannel=False,
        sym=True,
        mse=False,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
    ):
        self.maxq = torch.tensor(2**bit - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.bit = bit

    def find_params(self, x, weight=False, groupsize=-1):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        if groupsize != -1:
            # groupsize must be a divisor of infeatures
            assert weight is True and x.shape[1] % groupsize == 0
            groups = x.shape[1] // groupsize
        else:
            groups = 1

        shape = x.shape
        if self.perchannel:
            if weight:
                if groups > 1:
                    x = x.reshape((-1, groupsize))
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            if groups > 1:
                shape = [shape[0], groups] + [1] * (len(shape) - 1)
            else:
                shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


# Assumes layer is perfectly divisible into 1024 * 1024 blocks
class QuantLinear(nn.Module):
    def __init__(self, infeatures, outfeatures, bit=4, groupsize=-1):
        super().__init__()
        # 3 int32 solved at the same time for 3bit
        # 1 int32 solved at the same time for 4bit
        if groupsize != -1:
            min_group_size = {4: 128, 3: 128, 2: 64}[bit]
            assert groupsize % min_group_size == 0
            assert infeatures % groupsize == 0
            groups = infeatures // groupsize
        else:
            groups = 1
        self.infeatures = infeatures
        self.outfeatures = outfeatures

        self.groups = groups
        self.groupsize = groupsize

        if self.groups > 1:
            self.register_buffer("zeros", torch.zeros((outfeatures, self.groups, 1)))
            self.register_buffer("scales", torch.zeros((outfeatures, self.groups, 1)))
        else:
            self.register_buffer("zeros", torch.zeros((outfeatures, 1)))
            self.register_buffer("scales", torch.zeros((outfeatures, 1)))
        self.register_buffer("bias", torch.zeros(outfeatures))
        self.bit = bit
        self.parallel_bit_nums = 1 if self.bit in [2, 4] else 3
        assert self.bit in [2, 3, 4], "only support 2/3/4 bit now"
        self.register_buffer(
            "qweight",
            torch.zeros(
                (
                    ceiling_div(infeatures * self.bit, 32 * self.parallel_bit_nums)
                    * self.parallel_bit_nums,
                    outfeatures,
                ),
                dtype=torch.int,
            ),
        )

    def pack(self, linear, scales, zeros):
        self.zeros = zeros * scales
        self.scales = scales.clone()
        self.bias = linear.bias.clone()

        weight = linear.weight.data
        if self.groups > 1:
            weight = weight.view(self.outfeatures, self.groups, -1)
            intweight = torch.round((weight + self.zeros) / self.scales).to(torch.int)
            intweight = intweight.view(self.outfeatures, self.infeatures)
        else:
            intweight = torch.round((weight + self.zeros) / self.scales).to(torch.int)

        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros(
            (
                ceiling_div(intweight.shape[0] * self.bit, 32 * self.parallel_bit_nums)
                * self.parallel_bit_nums,
                intweight.shape[1],
            ),
            dtype=np.uint32,
        )
        i = 0
        row = 0
        H, W = intweight.shape
        while row < qweight.shape[0] and i < H:
            if self.bit == 2:
                for j in range(i, i + 16):
                    if j < H:
                        qweight[row] |= intweight[j] << (2 * (j - i))
                i += 16
                row += 1
            if self.bit == 4:
                for j in range(i, i + 8):
                    if j < H:
                        qweight[row] |= intweight[j] << (4 * (j - i))
                i += 8
                row += 1
            elif self.bit == 3:
                for j in range(i, i + 10):
                    if j < H:
                        qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10

                if i < H:
                    qweight[row] |= intweight[i] << 30
                    row += 1
                    qweight[row] |= (intweight[i] >> 2) & 1
                    i += 1
                else:
                    break

                for j in range(i, i + 10):
                    if j < H:
                        qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10

                if i < H:
                    qweight[row] |= intweight[i] << 31
                    row += 1
                    qweight[row] |= (intweight[i] >> 1) & 0x3
                    i += 1
                else:
                    break

                for j in range(i, i + 10):
                    if j < H:
                        qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

    def forward(self, x):
        f32 = lambda x: x.to(torch.float32)

        # prefer 32bit data as FC input
        # because 4w16f cuda implementation is 4x ~ 5x slower than 4w32f (2080ti)
        return (
            {2: Quant2Matmul, 3: Quant3Matmul, 4: Quant4Matmul}[self.bit]
            .apply(
                f32(x),
                self.qweight,
                f32(self.scales),
                f32(self.zeros),
                f32(self.bias),
                self.groupsize,
            )
            .to(x.dtype)
        )


class Quant4Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, qweight, scales, zeros, bias, groupsize=-1):
        x_shape = list(input.shape)
        y = (
            bias.to(input.dtype)[(None,) * (len(x_shape) - 1)]
            .repeat(x_shape[:-1] + [1])
            .contiguous()
        )
        if not input.is_cuda:
            input = input.cuda()
            qweight = qweight.cuda()
            y = y.cuda()
            scales = scales.cuda()
            zeros = zeros.cuda()
            is_cuda = False
        else:
            is_cuda = True
        if groupsize == -1:
            cuda_kernel.vecquant4matmul(input, qweight, y, scales, zeros)
        else:
            cuda_kernel.vecgroupquant4matmul(
                input, qweight, y, scales, zeros, groupsize
            )
        if not is_cuda:
            y = y.cpu()
        return y

    @staticmethod
    def backward(ctx, grad):
        return [None] * 6

    @staticmethod
    def symbolic(g, input, qweight, scales, zeros, bias, groupsize=-1):
        from torch.onnx.symbolic_helper import _get_tensor_sizes, _get_tensor_dim_size

        op_name = "GPTQ::Quant4Matmul"
        input_sizes = _get_tensor_sizes(input)
        opr_type = input.type().with_sizes(
            input_sizes[:-1] + [_get_tensor_dim_size(qweight, 1)]
        )
        args = (input, qweight, scales, zeros, bias)
        kwargs = {"groupsize_i": groupsize}
        opr = g.op(op_name, *args, **kwargs).setType(opr_type)
        return opr


class Quant3Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, qweight, scales, zeros, bias, groupsize=-1):
        x_shape = list(input.shape)
        y = (
            bias.to(input.dtype)[(None,) * (len(x_shape) - 1)]
            .repeat(x_shape[:-1] + [1])
            .contiguous()
        )
        if not input.is_cuda:
            input = input.cuda()
            qweight = qweight.cuda()
            y = y.cuda()
            scales = scales.cuda()
            zeros = zeros.cuda()
            is_cuda = False
        else:
            is_cuda = True
        if groupsize == -1:
            cuda_kernel.vecquant3matmul(input, qweight, y, scales, zeros)
        else:
            cuda_kernel.vecgroupquant3matmul(
                input, qweight, y, scales, zeros, groupsize
            )
        if not is_cuda:
            y = y.cpu()
        return y

    @staticmethod
    def backward(ctx, grad):
        return [None] * 6

    @staticmethod
    def symbolic(g, input, qweight, scales, zeros, bias, groupsize=-1):
        from torch.onnx.symbolic_helper import _get_tensor_sizes, _get_tensor_dim_size

        op_name = "GPTQ::Quant3Matmul"
        input_sizes = _get_tensor_sizes(input)
        opr_type = input.type().with_sizes(
            input_sizes[:-1] + [_get_tensor_dim_size(qweight, 1)]
        )
        args = (input, qweight, scales, zeros, bias)
        kwargs = {"groupsize_i": groupsize}
        opr = g.op(op_name, *args, **kwargs).setType(opr_type)
        return opr


class Quant2Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, qweight, scales, zeros, bias, groupsize=-1):
        x_shape = list(input.shape)
        y = (
            bias.to(input.dtype)[(None,) * (len(x_shape) - 1)]
            .repeat(x_shape[:-1] + [1])
            .contiguous()
        )
        if not input.is_cuda:
            input = input.cuda()
            qweight = qweight.cuda()
            y = y.cuda()
            scales = scales.cuda()
            zeros = zeros.cuda()
            is_cuda = False
        else:
            is_cuda = True
        if groupsize == -1:
            cuda_kernel.vecquant2matmul(input, qweight, y, scales, zeros)
        else:
            cuda_kernel.vecgroupquant2matmul(
                input, qweight, y, scales, zeros, groupsize
            )
        if not is_cuda:
            y = y.cpu()
        return y

    @staticmethod
    def backward(ctx, grad):
        return [None] * 6

    @staticmethod
    def symbolic(g, input, qweight, scales, zeros, bias, groupsize=-1):
        from torch.onnx.symbolic_helper import _get_tensor_sizes, _get_tensor_dim_size

        op_name = "GPTQ::Quant2Matmul"
        input_sizes = _get_tensor_sizes(input)
        opr_type = input.type().with_sizes(
            input_sizes[:-1] + [_get_tensor_dim_size(qweight, 1)]
        )
        args = (input, qweight, scales, zeros, bias)
        kwargs = {"groupsize_i": groupsize}
        opr = g.op(op_name, *args, **kwargs).setType(opr_type)
        return opr


def make_quant(module, layers_bit, name="", groupsize=-1):
    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + "." + attr if name != "" else attr
        if name1 in layers_bit:
            setattr(
                module,
                attr,
                QuantLinear(
                    tmp.in_features,
                    tmp.out_features,
                    bit=layers_bit[name1],
                    groupsize=groupsize,
                ),
            )
    for name1, child in module.named_children():
        make_quant(
            child,
            layers_bit,
            name + "." + name1 if name != "" else name1,
            groupsize=groupsize,
        )
