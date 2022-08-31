import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsebit.quantization.modules import QuantOpr, register_qmodule


@register_qmodule(sources=[nn.Upsample])
class QUpsample(QuantOpr):
    def __init__(self, org_module=None, config=None):
        super(QUpsample, self).__init__()
        self.scale_factor = org_module.scale_factor
        self.mode = org_module.mode
        self._repr_info = "QUpsample"

    def forward(self, x_in, *args):
        x_in = self.input_quantizer(x_in)
        out = F.interpolate(x_in, scale_factor=self.scale_factor, mode=self.mode)
        return out
