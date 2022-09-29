import torch

from sparsebit.sparse.sparsers import Sparser as BaseSparser
from sparsebit.sparse.sparsers import register_sparser


@register_sparser
class Sparser(BaseSparser):
    STRATEGY = "l1norm"

    def __init__(self, config, opr):
        super(Sparser, self).__init__(config, opr)
        self.ratio = config.SPARSER.RATIO
        self.opr = opr

    def calc_mask(self, x):
        data = x.detach()
        data = torch.abs(data).flatten()
        sorted_data, indices = torch.sort(data)
        thresh_idx = int(data.numel() * self.ratio)
        thresh_idx = min(thresh_idx, data.numel() - 1)
        thresh = sorted_data[thresh_idx]
        mask = data > thresh
        mask = mask.reshape(x.shape)

        return mask
