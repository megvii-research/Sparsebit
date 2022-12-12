import torch.nn as nn
import timm.models.layers as layers

from sparsebit.quantization.complicated_modules import register_complicated_module


@register_complicated_module(sources=[layers.norm_act.BatchNormAct2d])
class BatchNormAct2d(nn.Module):
    def __init__(self, org_module=None):
        super().__init__()
        try:
            self.bn = nn.BatchNorm2d(
                num_features=org_module.num_features,
                eps=org_module.eps,
                momentum=org_module.momentum,
                affine=org_module.affine,
                track_running_stats=org_module.track_running_stats,
                device=org_module.device,
                dtype=org_module.dtype,
            )
        except:
            self.bn = nn.BatchNorm2d(
                num_features=org_module.num_features,
                eps=org_module.eps,
                momentum=org_module.momentum,
                affine=org_module.affine,
                track_running_stats=org_module.track_running_stats,
            )
        self.bn.weight = org_module.weight
        self.bn.bias = org_module.bias
        self.bn.running_mean = org_module.running_mean
        self.bn.running_var = org_module.running_var
        self.bn.num_batches_tracked = org_module.num_batches_tracked
        self.drop = org_module.drop
        self.act = org_module.act

    def forward(self, x):
        x = self.bn(x)
        x = self.drop(x)
        x = self.act(x)
        return x
