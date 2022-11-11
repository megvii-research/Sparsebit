import torch
from typing import Callable
from sparsebit.quantization.converters.utils import ReplacePatternBase, MatchingNode
from sparsebit.quantization.modules import (
    QAdd,
    QReLU,
    QReLU6,
    QLeakyReLU,
    QMish,
    QConv2d,
    QLinear,
    QSigmoid,
    QBatchNorm2d,
)


def check(node, module):
    """检查是否已经关闭量化,防止死循环。用于ops的checker。

    Args:
        node (torch.fx.Node): 要匹配的node。
        module (torch.nn.Module): node对应的Module。
    """
    return not module.fake_fused


class ReplacePattern_DisableQuant(ReplacePatternBase):
    """在遇到特定结构时，关闭除首层以外其他层的量化,即调用 QuantOpr.set_fake_fused() 。

    如:

    - conv + bn
    - conv + relu
    - linear + bn
    - linear + relu

    等等。

    Args:
        matcher_ops (List[MatcherNode]):
            指定一个结构。除首层(入度为0的层)以外的其他层都会关闭量化。

    .. Note::

        在config.SCHEDULE配置变换的开关,默认为开启。
    """

    def __init__(self, matcher_ops):
        self.matcher_ops = matcher_ops
        super(ReplacePattern_DisableQuant, self).__init__()

    def make_nodes(self):
        """self.matcher_ops"""
        return self.matcher_ops

    def get_new_graph(self, nodes_dict, modules_dict, model=None, transform_idx=None):
        """自动识别和执行set_fake_fused()。"""
        noninput_node_names = set(
            [
                matcher_node.name
                for matcher_node in self.matcher_ops
                if not all(i is None for i in matcher_node.inputs)
            ]
        )
        anchor_node_name = set(i.name for i in self.matcher_ops) - set(
            inp_name for i in self.matcher_ops for inp_name in i.inputs
        )
        assert len(anchor_node_name) == 1
        anchor_node_name = list(anchor_node_name)[0]

        for noninput_node in noninput_node_names:
            op = modules_dict[noninput_node]
            op.set_fake_fused()
        return {anchor_node_name: nodes_dict[anchor_node_name]}


def make_chain_connection(op_types):
    """自动生成ops。

    只支持链式连接关系,即:一串op按顺序相连,除第一个op外其他op只能有唯一的输入。

    如果不是链式连接,则不能调用本函数。需要构造一个正确的ops。
    """
    nodes = []
    for idx, op_type in enumerate(op_types):
        if issubclass(op_type, torch.nn.Module):
            input_nums = op_type.forward.__code__.co_argcount - 1  # except "self" args
        elif isinstance(op_type, Callable):
            input_nums = op_type.__code__.co_argcount
        else:
            raise NotImplementedError("can't recognize class {}".format(op_type))
        if idx != 0:
            assert input_nums == 1
        nodes.append(
            MatchingNode(
                name="op_{}".format(idx),
                inputs=[None] * input_nums if idx == 0 else ["op_{}".format(idx - 1)],
                op_type=[op_type],
                checker=check if idx != 0 else (lambda op, module: True),
            )
        )
    return nodes


ReplacePatterns = [
    ReplacePattern_DisableQuant(make_chain_connection([QConv2d, QBatchNorm2d])),
    ReplacePattern_DisableQuant(make_chain_connection([QConv2d, QReLU])),
    ReplacePattern_DisableQuant(make_chain_connection([QConv2d, QReLU6])),
    ReplacePattern_DisableQuant(make_chain_connection([QConv2d, QSigmoid])),
    ReplacePattern_DisableQuant(make_chain_connection([QConv2d, QLeakyReLU])),
    ReplacePattern_DisableQuant(make_chain_connection([QConv2d, QMish])),
    ReplacePattern_DisableQuant(make_chain_connection([QLinear, QBatchNorm2d])),
    ReplacePattern_DisableQuant(make_chain_connection([QLinear, QReLU])),
    ReplacePattern_DisableQuant(make_chain_connection([QLinear, QReLU6])),
    ReplacePattern_DisableQuant(make_chain_connection([QLinear, QSigmoid])),
    ReplacePattern_DisableQuant(make_chain_connection([QBatchNorm2d, QReLU])),
    ReplacePattern_DisableQuant(make_chain_connection([QBatchNorm2d, QReLU6])),
    ReplacePattern_DisableQuant(make_chain_connection([QBatchNorm2d, QLeakyReLU])),
    ReplacePattern_DisableQuant(make_chain_connection([QBatchNorm2d, QMish])),
    ReplacePattern_DisableQuant(make_chain_connection([QAdd, QReLU])),
    ReplacePattern_DisableQuant(make_chain_connection([QAdd, QReLU6])),
]
