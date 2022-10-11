import torch

from sparsebit.sparse.sparsers import Sparser as BaseSparser
from sparsebit.sparse.sparsers import register_sparser


@register_sparser
class Sparser(BaseSparser):
    STRATEGY = "l1norm"

    def __init__(self, config, opr):
        super(Sparser, self).__init__(config, opr)

    def calc_mask(self, x):
        if self.ratio == 0.0:
            return torch.ones_like(x)

        if self.type == "unstructed":
            data = x.detach()
            data = torch.abs(data).flatten()
            sorted_data, indices = torch.sort(data)
            thresh_idx = int(data.numel() * self.ratio)
            thresh_idx = min(thresh_idx, data.numel() - 1)
            thresh = sorted_data[thresh_idx]
            mask = data > thresh
            mask = mask.reshape(x.shape)
        elif self.type == "structed":
            data = x.detach()
            data = torch.sum(torch.abs(data.reshape(data.shape[0], -1)), dim=1)
            sorted_data, indices = torch.sort(data, dim=0)
            pruned_channels = int(x.shape[0] * self.ratio)
            pruned_indices = indices[:pruned_channels].tolist()

            mask = torch.ones_like(x)
            for idx in pruned_indices:
                mask[idx] = torch.zeros_like(
                    torch.index_select(
                        data.cpu().detach(), dim=0, index=torch.tensor(idx)
                    )
                )

        return mask
