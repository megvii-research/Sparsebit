import torch
import torch.nn as nn
import numpy as np
from sparsebit.quantization.quantizers import Quantizer as BaseQuantizer
from sparsebit.quantization.quantizers import register_quantizer
from sparsebit.quantization.common import QuantTarget
from .quant_tensor import STE


@register_quantizer
class Quantizer(BaseQuantizer):
    TYPE = "adaround"

    def __init__(self, config):
        super(Quantizer, self).__init__(config)
        assert config.TARGET[0] == QuantTarget.WEIGHT, "AdaRound only supports to quant weights"
        self.zeta, self.gamma = 1.1, -0.1 # stretch-parameters

    def init_variables(self, x):
        x_floor = torch.floor(x / self.scale)
        rest = (x / self.scale) - x_floor
        v = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1) # => rectified_sigmoid(v)=rest
        self.v = nn.Parameter(v.to(x.device))

    def _get_soft_round_values(self):
        return torch.clamp(torch.sigmoid(self.v) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def _forward(self, x, scale, zero_point):
        x_floor = torch.floor(x / scale)
        if self.training:
            x_q = x_floor + self._get_soft_round_values()
        else: # evaluation
            x_q = x_floor + (self.v>=0).float()
        x_dq = STE.apply(x_q, torch.ones_like(scale), zero_point, self.qdesc, self.backend) * scale
        return x_dq


def reconstruct_qlayer(layer, inputs: torch.Tensor, outputs: torch.Tensor,
                       batch_size=32, max_steps=2000, beta_range=(20, 2),
                       warmup=0.0, p=2.0, round_loss_weight=1e-3):
    # init
    layer.set_quant(w_quant=True, a_quant=False)
    layer.weight_quantizer.init_variables(layer.weight)
    layer.weight_quantizer.train()
    opt_params = [layer.weight_quantizer.v]
    optimizer = torch.optim.Adam(opt_params)
    beta_decayer = LinearTempDecay(max_steps=max_steps, rel_start_step=warmup,
                                   start_beta=beta_range[0], end_beta=beta_range[1])
    loss_start_step = int(warmup * max_steps)
    print_freq = 1
    # training
    device = layer.weight.device
    for step in range(max_steps):
        idx = torch.randperm(inputs.size(0))[:batch_size]
        cur_input, cur_output = inputs[idx].to(device), outputs[idx].to(device)
        optimizer.zero_grad()
        quant_output = layer(cur_input)
        # calculate reconstruct_loss
        rec_loss = (quant_output - cur_output).abs().pow(p).sum(1).mean()
        # calculate round loss
        if step < loss_start_step:
            beta = round_loss = 0
        else:
            beta = beta_decayer(step)
            round_vals = layer.weight_quantizer._get_soft_round_values()
            round_loss = (1 - ((round_vals - .5).abs() * 2).pow(beta)).sum()
        loss = rec_loss + round_loss_weight * round_loss
        loss.backward()
        optimizer.step()
        if step % print_freq == 0:
            print('Loss: {:.3f} (rec: {:.3f}, round: {:.3f}) beta={:.2f} step={}'.format(loss, rec_loss, round_loss, beta, step))
    torch.cuda.empty_cache()
    layer.weight_quantizer.eval()


class LinearTempDecay:
    def __init__(self, max_steps, rel_start_step=0.2, start_beta=10, end_beta=2):
        self.max_steps = max_steps
        self.start_step = int(rel_start_step * max_steps)
        self.start_beta = start_beta
        self.end_beta = end_beta

    def __call__(self, step):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if step < self.start_step:
            return self.start_beta
        else:
            ratio = (step - self.start_step) / (self.max_steps - self.start_step)
            beta = (self.end_beta + (self.start_beta - self.end_beta) * max(0., (1-ratio)))
            return beta

