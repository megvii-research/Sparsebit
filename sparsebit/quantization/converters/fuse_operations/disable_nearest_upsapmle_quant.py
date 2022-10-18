import torch

from ..base import ReplacePatternBase, MatcherNode
from sparsebit.quantization.modules import QUpsample


def check_upsample(node, module):
    return not module.fake_fused and module.mode == "nearest"


class ReplacePattern(ReplacePatternBase):
    def __init__(self):
        super(ReplacePattern, self).__init__()

    def make_ops(self):
        return [
            MatcherNode(
                "upsample", inputs=[None], op_type=[QUpsample], checker=check_upsample
            ),
        ]

    def get_new_graph(self, nodes_dict, modules_dict, model=None, transform_idx=None):
        upsample_node = nodes_dict["upsample"]
        upsample_module = modules_dict["upsample"]
        upsample_module.set_fake_fused()
        return upsample_node
