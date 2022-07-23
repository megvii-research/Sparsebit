import torch

from ..base import ReplacePatternBase, MatcherNode


class ReplacePattern(ReplacePatternBase):
    def __init__(self):
        super(ReplacePattern, self).__init__()

    def make_ops(self):
        return [
            MatcherNode("identity", inputs=[None], op_type=[torch.nn.Identity]),
        ]

    def get_new_graph(self, nodes_dict, modules_dict, model=None, transform_idx=None):
        identity_node = nodes_dict["identity"]
        return identity_node.args[0]
