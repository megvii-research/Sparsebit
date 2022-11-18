import torch

from sparsebit.quantization.converters.utils import ReplacePatternBase, MatchingNode


def check_remove(node, module):
    flag = getattr(module, "remove", True)
    return flag

class ReplacePattern(ReplacePatternBase):
    """去掉网络中的identity算子"""

    def __init__(self):
        super(ReplacePattern, self).__init__()

    def make_nodes(self):
        """匹配identity op"""
        return [
            MatchingNode("identity", inputs=[None], checker = check_remove, op_type=[torch.nn.Identity]),
        ]

    def get_new_graph(self, nodes_dict, modules_dict, model=None, transform_idx=None):
        """移除identity op"""
        identity_node = nodes_dict["identity"]
        return {"identity": identity_node.args[0]}
