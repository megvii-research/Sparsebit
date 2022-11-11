import torch
import operator

from sparsebit.quantization.converters.utils import ReplacePatternBase, MatchingNode


class ReplacePattern(ReplacePatternBase):
    def __init__(self):
        super(ReplacePattern, self).__init__()

    def make_nodes(self):
        return [
            MatchingNode(
                "unbind", inputs=[None], op_type=[torch.unbind, torch.Tensor.unbind]
            ),
            MatchingNode("getitem", inputs=["unbind"], op_type=[operator.getitem]),
        ]

    def get_new_graph(self, nodes_dict, modules_dict, model=None, transform_idx=None):
        unbind_node = nodes_dict["unbind"]
        getitem_node = nodes_dict["getitem"]

        if isinstance(getitem_node.args[1], slice):  # w/o squeeze
            new_slice = getitem_node.args[1]
        else:  # w/ squeeze
            idx = int(getitem_node.args[1])
            new_slice = idx

        axis = unbind_node.args[1]
        new_subtensor_args = []
        for i in range(axis):
            new_subtensor_args.append(slice(None, None, None))
        new_subtensor_args.append(new_slice)
        new_subtensor_args = (
            tuple(new_subtensor_args)
            if len(new_subtensor_args) > 1
            else new_subtensor_args[0]
        )

        new_args = (unbind_node.args[0], new_subtensor_args)
        getitem_node.args = new_args

        return {"getitem": getitem_node}
