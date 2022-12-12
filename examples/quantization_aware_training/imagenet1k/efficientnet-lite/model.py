from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from timm.models.efficientnet_builder import (
    resolve_bn_args,
    decode_arch_def,
    round_channels,
    resolve_act_layer,
)
from timm.models.efficientnet import _create_effnet


def _gen_efficientnet_lite(
    arch,
    channel_multiplier=1.0,
    depth_multiplier=1.0,
    ckpt_path=None,
    num_classes=1000,
    **kwargs
):
    """Creates an EfficientNet-Lite model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
      'efficientnet-lite0': (1.0, 1.0, 224, 0.2),
      'efficientnet-lite1': (1.0, 1.1, 240, 0.2),
      'efficientnet-lite2': (1.1, 1.2, 260, 0.3),
      'efficientnet-lite3': (1.2, 1.4, 280, 0.3),
      'efficientnet-lite4': (1.4, 1.8, 300, 0.3),

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage
    """
    arch_def = [
        ["ds_r1_k3_s1_e1_c16"],
        ["ir_r2_k3_s2_e6_c24"],
        ["ir_r2_k5_s2_e6_c40"],
        ["ir_r3_k3_s2_e6_c80"],
        ["ir_r3_k5_s1_e6_c112"],
        ["ir_r4_k5_s2_e6_c192"],
        ["ir_r1_k3_s1_e6_c320"],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, fix_first_last=True),
        num_features=1280,
        stem_size=32,
        fix_stem=True,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        act_layer=resolve_act_layer(kwargs, "relu6"),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs,
    )
    model = _create_effnet(arch, pretrained=False, **model_kwargs)
    if ckpt_path:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
    return model


def efficientnet_lite0(num_classes=1000, ckpt_path=None):
    return _gen_efficientnet_lite(
        "efficientnet_lite0",
        channel_multiplier=1.0,
        depth_multiplier=1.0,
        ckpt_path=ckpt_path,
        num_classes=num_classes,
        bn_eps=1e-3,
        bn_momentum=0.1,
    )
