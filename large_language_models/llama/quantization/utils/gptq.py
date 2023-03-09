import math
import time

import torch
import torch.nn as nn
import transformers

from .quant import quantize


DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.mean_inp = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.mean_inp += inp.float()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        threshold=1e-3,
        bias_correction=True,
    ):
        weight = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            weight = weight.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            weight = weight.t()
        weight = weight.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        weight[:, dead] = 0
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for idx, quantizer in enumerate(self.quantizers):
            W = weight.clone()
            if not quantizer.ready():
                quantizer.find_params(W, weight=True)  # 先有个初始量化参数

            Losses = torch.zeros_like(W)
            Q = torch.zeros_like(W)

            for i1 in range(0, self.columns, blocksize):  # 以blocksize为单位进行处理
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                for i in range(count):  # 每一列做一次,
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if groupsize != -1:
                        if (i1 + i) % groupsize == 0:
                            quantizer.find_params(
                                W[:, (i1 + i) : (i1 + i + groupsize)], weight=True
                            )

                    q = quantize(  # 按列作量化, 但为什么scale/zero不重新统计呢? 是不是意味着更新不重要了?
                        w.unsqueeze(1), quantizer.scale, quantizer.zero, quantizer.maxq
                    ).flatten()
                    Q1[:, i] = q
                    Losses1[:, i] = (
                        w - q
                    ) ** 2 / d**2  # 误差大小, 平方是有道理的, 因为做了cholesky分解.

                    err1 = (w - q) / d  # 只有一列
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                Q[:, i1:i2] = Q1
                Losses[:, i1:i2] = Losses1 / 2

                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

                if DEBUG:
                    self.layer.weight.data[:, :i2] = Q[:, :i2]
                    self.layer.weight.data[:, i2:] = W[:, i2:]
                    print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                    print(torch.sum(Losses))
            torch.cuda.synchronize()
            mean_error = torch.mean(Losses).item()
            if mean_error < threshold:
                break
        print("time {:.2f}, mean-error {:.5f}".format(time.time() - tick, mean_error))
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        Q = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if bias_correction:
            delta_w = self.layer.weight.data - Q
            mean_inp = self.mean_inp.float() / self.nsamples
            mean_inp = mean_inp.mean(axis=1).unsqueeze(1)
            delta_bias = delta_w.float() @ mean_inp[:, 0]
            if self.layer.bias is not None:
                self.layer.bias.data += delta_bias
            else:
                self.layer.bias = nn.Parameter(delta_bias)
        self.layer.bias.data = self.layer.bias.data.to(torch.half)
        self.layer.weight.data = Q

        if DEBUG:
            print("MSE LOSS: ", torch.sum((self.layer(self.inp1) - self.out1) ** 2))
        return idx

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
