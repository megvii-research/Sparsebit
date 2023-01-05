import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from sparsebit.quantization.quantizers import Quantizer as BaseQuantizer
from sparsebit.quantization.quantizers import register_quantizer
from .quant_tensor import STE


@register_quantizer
class Quantizer(BaseQuantizer):
    TYPE = "Quadapter"

    def __init__(self, config):
        super(Quantizer, self).__init__(config)
        self.reconstruct_qlayer = reconstruct_qlayer

    def init_variables(self, x: torch.Tensor):
        alpha_shape = [1 for _ in range(self.dims)]
        alpha_shape[self.qdesc._ch_axis] = x.shape[self.qdesc._ch_axis]
        self.alpha = nn.Parameter(torch.ones(alpha_shape).to(self.device))

    def update_observer(self, x):
        self.dims = len(x.shape)
        self.observer.data_cache.update(x.detach())

    def _forward(self, x_f, scale, zero_point):
        x_f = x_f * self.alpha
        x_dq = STE.apply(x_f, scale, zero_point, self.qdesc, self.backend)
        x_dq = x_dq / self.alpha
        return x_dq


def reconstruct_qlayer(
    layer,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    batch_size=32,
    max_steps=20000,
    p=2.0,
):
    # init
    layer.eval()
    layer.set_quant(w_quant=True, a_quant=True)
    layer.input_quantizer.init_variables(inputs)
    layer.input_quantizer.train()
    opt_params = [layer.input_quantizer.alpha]
    optimizer = torch.optim.Adam(opt_params)
    print_freq = 500
    # training
    device = layer.input_quantizer.device
    inputs, outputs = inputs.to(device), outputs.to(device)
    for step in range(max_steps):
        idx = torch.randperm(inputs.size(0))[:batch_size]
        cur_input, cur_output = inputs[idx], outputs[idx]
        optimizer.zero_grad()
        quant_output = layer(cur_input)
        loss = (quant_output - cur_output).abs().pow(p).sum(1).mean()
        loss.backward(retain_graph=True)
        optimizer.step()
        if step % print_freq == 0:
            print("Loss: {:.3f}  step={}".format(loss, step))
    torch.cuda.empty_cache()
    layer.input_quantizer.eval()
