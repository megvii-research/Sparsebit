import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval

from ..base import ReplacePatternBase, MatcherNode
from sparsebit.quantization.modules import QConv2d, QLinear, QBatchNorm2d


class ReplacePattern(ReplacePatternBase):
    """把conv-bn或linear-bn吸成一整个新conv / linear。

    .. Note::

        在config.SCHEDULE配置变换的开关,默认为关闭。
    """

    def __init__(self):
        super(ReplacePattern, self).__init__()

    def make_ops(self):
        """匹配 conv-bn / linear-bn 结构。"""
        return [
            MatcherNode(
                "cnn_layer",
                inputs=[None],
                op_type=[QConv2d, QLinear],
            ),
            MatcherNode(
                "bn",
                inputs=["cnn_layer"],
                op_type=[nn.BatchNorm2d, QBatchNorm2d],
            ),
        ]

    def get_new_graph(self, nodes_dict, modules_dict, model, transform_idx):
        """替换子图部分调用了 ``torch.nn.utils.fusion`` api

        - fuse_conv_bn_eval

        - fuse_linear_bn_eval
        """
        cnn_module = modules_dict["cnn_layer"]
        bn_module = modules_dict["bn"].module
        module_training = cnn_module.training
        cnn_module.eval()
        bn_module.eval()
        cnn_node = nodes_dict["cnn_layer"]
        if isinstance(cnn_module, QConv2d):
            new_cnn_module = fuse_conv_bn_eval(cnn_module, bn_module)
            op_name = cnn_node.target + "_bn"
        else:
            new_cnn_module = fuse_linear_bn_eval(cnn_module, bn_module)
            op_name = cnn_node.target + "_bn"
        # new_cnn_module is QuantOpr copied from cnn_module with new parameters
        # see torch.nn.utils.fusion.fuse_conv_bn_eval / fuse_linear_bn_eval
        model.add_module(op_name, new_cnn_module)
        if module_training:
            new_cnn_module.train()

        (x_in,) = cnn_node.args
        with model.graph.inserting_after(cnn_node):
            new_node = model.graph.create_node(
                op="call_module",
                target=op_name,
                args=(x_in,),
                name=op_name,
            )
        return new_node
