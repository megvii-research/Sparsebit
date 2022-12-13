import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsebit.quantization.modules import QuantOpr, register_qmodule


@register_qmodule(sources=[nn.Embedding])
class QEmbedding(QuantOpr):
    """量化嵌入层, 仅有 ``weight_quantizer``, 默认由于输入是index值, 即不量化输入.

    是QuantOpr的子类。

    Attributes:
        weight_quantizer (sparsebit.quantization.quantizers.base.Quantizer):
        weight (torch.nn.Parameter): embedding的weight,引用自原Module。
    """

    def __init__(self, org_module, config=None):
        assert isinstance(org_module, nn.Embedding)
        super().__init__()
        self.cfg = config
        self.weight = org_module.weight
        self.padding_idx = org_module.padding_idx
        self.max_norm = org_module.max_norm
        self.norm_type = org_module.norm_type
        self.scale_grad_by_freq = org_module.scale_grad_by_freq
        self.sparse = org_module.sparse
        self._repr_info = "Q" + org_module.__repr__()

    def build_quantizer(self, config):
        QuantOpr.build_quantizer(self, config)
        self.input_quantizer.set_fake_fused()

    def forward(self, x_in, *args, **kwargs):
        weight = self.weight_quantizer(self.weight)
        return F.embedding(
            x_in, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
