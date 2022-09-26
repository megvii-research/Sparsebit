import torch
from yacs.config import CfgNode as CN
from sparsebit.utils.yaml_utils import _parse_config, update_config

_C = CN()

_C.PRUNER = CN()
_C.PRUNER.TYPE = ""  # support structed / unstructed
_C.PRUNER.STRATEGY = ""  # l1norm / slimming
_C.PRUNER.GRANULARITY = ""  # support layerwise / channelwise
_C.PRUNER.RATIO = 0.0
_C.PRUNER.SPECIFIC = []


def parse_pconfig(cfg_file):
    pconfig = _parse_config(cfg_file, default_cfg=_C)
    return pconfig
