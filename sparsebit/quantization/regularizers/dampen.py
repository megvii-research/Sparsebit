import torch
import math
from sparsebit.quantization.regularizers import Regularizer as BaseRegularizer
from sparsebit.quantization.regularizers import register_regularizer


@register_regularizer
class Regularizer(BaseRegularizer):
    TYPE = "Dampen"

    def __init__(self, config):
        super(Regularizer, self).__init__(config)
        self.config = config
        self._lambda = config.REGULARIZER.LAMBDA
        self.iter = 0
        self.total_iter = 46344

    def _get_loss(self, x, quantizer):

        x_q = quantizer(x)

        qmin, qmax = quantizer.qdesc.qrange

        scale, zero_point = quantizer._qparams_preprocess(x)

        scale = scale.detach()
        zero_point = zero_point.detach()

        min_val = (qmin - zero_point) * scale

        max_val = (qmax - zero_point) * scale

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
                coeff = (self.iter  + 1) / self.total_iter
        schedule_value = 1 - 0.5 * (1.0 + math.cos(math.pi * coeff))
        self.iter += 1
        print(self.iter)
        return loss * self._lambda * schedule_value