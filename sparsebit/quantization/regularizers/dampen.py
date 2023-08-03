import torch

from sparsebit.quantization.regularizers import Regularizer as BaseRegularizer
from sparsebit.quantization.regularizers import register_regularizer
from sparsebit.quantization.quantizers.quant_tensor import fake_qrange_factory


@register_regularizer
class Regularizer(BaseRegularizer):
    TYPE = "Dampen"

    def __init__(self, config):
        super(Regularizer, self).__init__(config)
        self.config = config

    def _get_loss(self, x, quantizer):

        x_q = quantizer(x)

        scale, zero_point = quantizer._qparams_preprocess(x)

        min_val, max_val = fake_qrange_factory[quantizer.backend](
            scale, zero_point, quantizer.qdesc
        )

        min_val = min_val.detach()
        max_val = max_val.detach()

        x_c = torch.min(torch.max(x, min_val), max_val)

        loss = (x_q - x_c) ** 2

        loss = loss.sum()

        return loss

    def __call__(self, model):
        loss = 0.0
        for n, m in model.named_modules():
            if (
                hasattr(m, "weight")
                and hasattr(m, "weight_quantizer")
                and m.weight_quantizer
                and m.weight_quantizer.is_enable
            ):
                loss += self._get_loss(m.weight, m.weight_quantizer)
        return loss
