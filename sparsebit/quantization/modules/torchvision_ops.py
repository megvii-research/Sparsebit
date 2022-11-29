import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.stochastic_depth import stochastic_depth
from functools import partial
from sparsebit.quantization.modules import QuantOpr, register_qmodule


@register_qmodule(sources=[stochastic_depth])
class StochasticDepth(nn.Module):
    def __init__(self, org_module, config=None):
        super().__init__()
        # if isinstance(org_module, torch.fx.Node):
        #    self.p = org_module.args[1]
        #    self.mode = org_module.args[2]
        # else:
        #    raise NotImplementedError

    def forward(self, x_in, *args, **kwargs):
        out = stochastic_depth(x_in, *args, **kwargs)
        return out
