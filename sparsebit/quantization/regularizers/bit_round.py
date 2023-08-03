import torch
import math
from sparsebit.quantization.regularizers import Regularizer as BaseRegularizer
from sparsebit.quantization.regularizers import register_regularizer
from sparsebit.quantization.quantizers.base import Quantizer


@register_regularizer
class Regularizer(BaseRegularizer):
    TYPE = "bit_round"

    def __init__(self,
        config,
        qmodel,
        max_coeff = 1e6,
        epochs_wo_regularization = 20,
        total_epochs = 90,
    ):
        super(Regularizer, self).__init__(config)
        self.epochs_wo_regularization = epochs_wo_regularization
        self.total_epochs = total_epochs - epochs_wo_regularization
        self.config = config
        self.max_coeff = max_coeff
        self.quantizers = []
        for m in qmodel.modules():
            if isinstance(m, Quantizer) and not m.fake_fused:
                self.quantizers.append(m)

    def __call__(self, epoch):
        loss = torch.tensor(0)
        if epoch < self.epochs_wo_regularization:
            return loss
        for quantizer in self.quantizers:
            bit = torch.sqrt(2*quantizer.qmax+2) if quantizer.qdesc.is_symmetric else torch.sqrt(quantizer.qmax+1)
            bit_floor = math.floor(bit.item())
            bit_bias = bit - bit_floor
            loss += bit_bias*(1-bit_bias)/bit_floor
        coeff = self.max_coeff*(epoch-self.epochs_wo_regularization+1)/self.total_epochs
        return coeff*loss/len(self.quantizers)