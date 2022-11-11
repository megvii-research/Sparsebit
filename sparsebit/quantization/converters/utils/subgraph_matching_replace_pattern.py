from typing import List, Dict, Tuple, Callable
import torch
import torch.fx

from .subgraph_matching_node import MatchingNode
from .subgraph_matching_utils import (
    SubgraphMatchingTimer,
    ReplaceStrategyBase,
    ReplaceStrategy,
)
from .subgraph_matching import SubgraphMatcher
from ..prune import PruneGraph


class ReplacePatternBase(object):
    """子图替换的基类。应当由各种不同的变换继承。
    Args:
        nodes (List[MatcherNode]):
            构建好的子图,需要在初始化时由 ``self.get_new_graph`` 生成,传入SubgraphMatcher中。
        joint_checkers (List[Tuple[Tuple[str], Callable]]):
            多个MatcherNode的联合自定义匹配条件 list。
            在初始化时由 ``self.make_joint_checkers`` 生成,传入SubgraphMatcher中。
        matching_strategy (ReplaceStrategy):
            子图替换使用的策略。在初始化时由 ``self.make_matching_strategy`` 生成。
    .. Note::
        每次替换后都会对网络做剪枝,删除与输出无关的算子。
    """

    def __init__(self):
        self.nodes = self.make_nodes()
        self.joint_checkers = self.make_joint_checkers()
        self.matching_strategy = self.make_matching_strategy()
        self.timer = SubgraphMatchingTimer()

    def get_new_graph(
        self,
        nodes_dict: Dict[str, torch.fx.Node],
        modules_dict: Dict[str, torch.nn.Module],
        model: torch.fx.GraphModule,
        transform_idx: int,
    ):
        """需要重载的方法。输入{op_name: torch.fx.Node}和{op_name: object}的映射,构造一个变换后的新子图。
        只需要返回一个node anchor,即matcher_nodes子图唯一的输出点。
        请注意只有在node对应modules是一个引用的时候修改才是有意义的,单个int等object没有修改的意义。
        """
        raise NotImplementedError("graph modification not implemented")

    def make_nodes(self) -> List[MatchingNode]:
        """需要重载的方法,指定一个要匹配的子图。"""
        raise NotImplementedError("replace_pattern nodes not implemented")

    def make_joint_checkers(self) -> List[Tuple[Tuple[str], Callable]]:
        """需要重载的方法,指定需要多个op和对应modules_list一起判断的条件,
        module和前面op名称一一对应。默认不使用任何联合checker。
        .. Note::
            一个检查2 concats所在轴相等的示例如下::
            >>> [
            >>>     (
            >>>         ("cat1", "cat2"),
            >>>         lambda cat1, cat2, modules: modules["cat1"].axis == modules["cat2"].axis
            >>>     )
            >>> ]
        """
        return []

    def make_matching_strategy(self) -> ReplaceStrategyBase:
        """可以重载的方法，指定匹配策略。
        默认使用 ``ReplaceStrategy.APPLY_REPEAT`` 。
        """
        return ReplaceStrategy.APPLY_REPEAT

    def apply(self, m: torch.fx.GraphModule) -> bool:
        """执行子图替换。"""
        flg = True
        is_transformed = False
        while flg:
            result = self.apply_once(m)
            is_transformed |= result
            flg = self.matching_strategy.repeat & result
        return is_transformed

    def apply_once(self, m: torch.fx.GraphModule) -> bool:
        """执行单次子图替换。"""

        def replace_op(
            nodes: List[torch.fx.Node],
            origin_op: torch.fx.Node,
            target_op: torch.fx.Node,
        ):
            for node in nodes:
                if node != origin_op:
                    node.replace_input_with(origin_op, target_op)

        matcher = SubgraphMatcher(
            matching_nodes=self.nodes,
            joint_checkers=self.joint_checkers,
            matching_strategy=self.matching_strategy,
        )
        nodes_dict, modules_dict = matcher.apply(m)
        is_transformed = nodes_dict is not None and modules_dict is not None
        if is_transformed:
            nodes = list(m.graph.nodes)
            replace_dict = self.get_new_graph(
                nodes_dict=nodes_dict,
                modules_dict=modules_dict,
                model=m,
                transform_idx=self.timer.get_idx(),
            )
            for rep_name, new_node in replace_dict.items():
                replace_op(nodes, nodes_dict[rep_name], new_node)
        m = PruneGraph().apply(m)
        return is_transformed
