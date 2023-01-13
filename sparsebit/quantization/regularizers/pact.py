import torch

from sparsebit.quantization.regularizers import Regularizer as BaseRegularizer
from sparsebit.quantization.regularizers import register_regularizer


@register_regularizer
class Regularizer(BaseRegularizer):
    TYPE = "Pact"

    def __init__(self, config):
        super(Regularizer, self).__init__(config)
        self.config = config

    def __call__(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            if "alpha" in n:
                loss += (p**2).sum()
        return loss