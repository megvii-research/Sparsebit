import operator

from ..base import ReplacePatternBase, MatcherNode
from sparsebit.quantization.modules.shape import Size


class ReplacePattern(ReplacePatternBase):
    """把网络出现的 getattr-getitem 得到shape的结构直接改成Size算子"""

    def __init__(self):
        super(ReplacePattern, self).__init__()

    def make_ops(self):
        """匹配

        - getattr("shape")

        - getitem(dim)
        """
        return [
            MatcherNode(
                "getattr",
                inputs=[None],
                op_type=[getattr],
                checker=lambda node, module: node.args[1] == "shape",
            ),
            MatcherNode(
                "getitem",
                inputs=["getattr"],
                op_type=[operator.getitem],
            ),
        ]

    def get_new_graph(self, nodes_dict, modules_dict, model, transform_idx):
        """替换成Size算子"""
        getitem_node = nodes_dict["getitem"]
        getattr_node = nodes_dict["getattr"]
        op_name = "size_{}".format(transform_idx) if transform_idx else "size"

        # HACK: sending getattr node because args match
        # getattr_node.args = (input, idx)
        # size_node.args = (input, dim) | (input,)
        new_module = Size(getitem_node)
        model.add_module(op_name, new_module)

        old_input = getattr_node.args[0]
        with model.graph.inserting_after(getattr_node):
            new_node = model.graph.create_node(
                op="call_module",
                target=op_name,
                args=(old_input,),
                kwargs={"dim": getitem_node.args[1]},
                name=op_name,
            )
        return new_node
