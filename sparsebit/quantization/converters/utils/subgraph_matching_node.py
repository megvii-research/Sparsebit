from typing import List, Union, Callable
from enum import Enum


class InputMatchingType(Enum):
    """在子图替换中,可以使用的MatchingNode输入匹配类型。

    Args:
        ALL: 完全按照MatchingNode给出的顺序匹配torch.fx.Node的输入。
        SUBSET:
            允许MatchingNode给出的输入是torch.fx.Node的输入的子集。
            即允许任意交换MatchingNode的输入顺序,并允许补None。

    .. Warning::
        目前SUBSET功能未实装。

    """

    ALL = 0
    SUBSET = 1


class MatchingNode(object):
    """Matching中的node匹配基类。

    Args:
        name (str):
            在一个Matching中不可重复。用于在Matching子图中标识每个node。
        inputs (List[Union[str, None]]):
            该点的输入节点名,在Matching子图中表示连接关系。

            如果没有输入，留空: ``[]`` 。

            如果有输入的连接关系但不希望放在匹配里或者写成一个新MatchingNode,使用 ``None`` 。
        op_type (List[Union[Callable, object]]):
            一个允许类型的list,表示这个op可以是哪些类型。
        checker (Callable):
            单个op (与对应module)的自定义匹配条件,输入(node_op, node_module),输出bool。

            .. Note::

                一个示例如下::

                >>> lambda cat, module: module.axis == 1

        input_match_type (int):
            输入一个InputMatchType,在Matching子图中表示 ``inputs`` 的匹配要求。
            参见 (InputMatchType) 。

    在以下条件满足时,可以匹配graph中的一个node:

    - op_type对齐。

    - checker中的条件满足。

    如果需要匹配子图,那么还需要满足 ``inputs`` 和 ``input_match_type`` 规定的子图连接关系限制。

    """

    def __init__(
        self,
        name: str,
        inputs: List[Union[str, None]],
        op_type: List[Union[Callable, object]],
        checker: Callable = lambda x, module: True,
        input_match_type=InputMatchingType.ALL,
    ):
        self.name = name
        self.inputs = inputs
        self.op_type = op_type
        self.checker = checker
        self.input_match_type = input_match_type

    def __repr__(self):
        return "MatchingNode(name={}, inputs={}, type={}, checker={}, match_type={})".format(
            self.name,
            self.inputs,
            self.op_type,
            self.checker,
            InputMatchingType(self.input_match_type).name,
        )
