import torch
from yacs.config import CfgNode as CN
from sparsebit.utils.yaml_utils import _parse_config, update_config

_C = CN()

_C.SPARSER = CN()
_C.SPARSER.TYPE = ""  # support structed / unstructed
_C.SPARSER.STRATEGY = ""  # l1norm / slimming
_C.SPARSER.RATIO = 0.0


def parse_sconfig(cfg_file):
    sconfig = _parse_config(cfg_file, default_cfg=_C)
    return sconfig
