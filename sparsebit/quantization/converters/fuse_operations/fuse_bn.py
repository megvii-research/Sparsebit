import torch
import copy
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval

from sparsebit.quantization.converters.utils import ReplacePatternBase, MatchingNode
from sparsebit.quantization.modules import QConv2d, QLinear, QBatchNorm2d


class ReplacePattern(ReplacePatternBase):
    """把conv-bn或linear-bn吸成一整个新conv / linear。

    .. Note::

        在config.SCHEDULE配置变换的开关,默认为关闭。
    """

    def __init__(self):
        super(ReplacePattern, self).__init__()

    def make_nodes(self):
        """匹配 conv-bn / linear-bn 结构。"""
        return [
            MatchingNode(
                "cnn_layer",
                inputs=[None],
                op_type=[QConv2d, QLinear],
            ),
            MatchingNode(
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

        # fuse bn under bn-tuning
        if cnn_module.weight_quantizer.is_enable:
            if isinstance(cnn_module, QConv2d):
                new_cnn_module = fuse_qconv_qbn(cnn_module, bn_module)
            else:
                new_cnn_module = fuse_qlinear_qbn(cnn_module, bn_module)
        else:
            if isinstance(cnn_module, QConv2d):
                new_cnn_module = fuse_conv_bn_eval(cnn_module, bn_module)
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
        return {"bn": new_node}


def fuse_qconv_qbn(cnn_module, bn_module):
    # if cnn_module.weight_quantizer.is_symmetric: # zp=0
    new_cnn_module = copy.deepcopy(cnn_module)
    bn_rm, bn_rv, bn_w, bn_b = (
        bn_module.running_mean,
        bn_module.running_var,
        bn_module.weight,
        bn_module.bias,
    )
    bn_rstd = torch.rsqrt(bn_rv + bn_module.eps)
    scale_ratio = (
        (bn_w * bn_rstd)
        .detach()
        .reshape([-1] + [1] * (len(cnn_module.weight.shape) - 1))
    )
    conv_w = new_cnn_module.weight * scale_ratio
    new_cnn_module.weight_quantizer.scale *= scale_ratio
    if cnn_module.bias is None:
        conv_b = torch.zeros_like(bn_rm)
    conv_b = (conv_b - bn_rm) * scale_ratio.reshape(-1) + bn_b
    new_cnn_module.weight = torch.nn.Parameter(conv_w)
    new_cnn_module.bias = torch.nn.Parameter(conv_b)
    return new_cnn_module


def fuse_qlinear_qbn(linear_module, bn_module):
    new_linear_module = copy.deepcopy(linear_module)
    bn_rm, bn_rv, bn_w, bn_b = (
        bn_module.running_mean,
        bn_module.running_var,
        bn_module.weight,
        bn_module.bias,
    )
    scale_ratio = (
        (bn_w * torch.rsqrt(bn_rv + bn_module.eps))
        .detach()
        .reshape([-1] + [1] * (len(linear_module.weight.shape) - 1))
    )
    new_linear_module.weight_quantizer.scale *= scale_ratio
    linear_w = new_linear_module.weight * scale_ratio
    if new_linear_module.bias is None:
        linear_b = torch.zeros_like(bn_rm)
    linear_b = (linear_b - bn_rm) * scale_ratio.reshape(-1) + bn_b
    new_linear_module.weight = torch.nn.Parameter(linear_w)
    new_linear_module.bias = torch.nn.Parameter(linear_b)
    return new_linear_module
