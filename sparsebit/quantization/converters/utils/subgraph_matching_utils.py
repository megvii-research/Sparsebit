from typing import Dict, List
import torch
import torch.fx


class SubgraphMatchingTimer(object):
    """一个计数器。对同个变换，多次调用能生成不同的序号。"""

    def __init__(self):
        self.idx = 0

    def get_idx(self):
        idx = self.idx
        self.idx += 1
        return idx


class ReplaceStrategyBase(object):
    """替换子图的替换策略基类。

    Args:
        repeat (bool): 是否要循环调用到网络中不再出现子图为止。
    """

    def __init__(self, repeat: bool):
        self.repeat = repeat


class ReplaceStrategy(object):
    """替换子图的替换策略。

    Args:

        APPLY_REPEAT (ReplaceStrategyBase):
            找到匹配子图并替换，循环到不再出现为止。
        APPLY_ONCE (ReplaceStrategyBase):
            找到匹配子图并替换，只执行一次。
    """

    APPLY_REPEAT = ReplaceStrategyBase(repeat=True)
    APPLY_ONCE = ReplaceStrategyBase(repeat=False)


def recursive_getattr(op, name, default_val=None):
    if name == "":
        return op
    pos = name.find(".")
    if pos == -1:
        return getattr(op, name, default_val)
    return recursive_getattr(getattr(op, name[:pos]), name[pos + 1 :], default_val)


def get_operators_type(
    ops: List[torch.fx.Node],
    m: torch.fx.GraphModule,
    named_modules: Dict[str, torch.fx.GraphModule],
):
    real_ops = []
    for op in ops:
        if op.op in ["placeholder", "output"]:  # input / output
            real_op = None
        elif op.op == "get_attr":  # parameter or constants
            real_op = recursive_getattr(m, op.target).__class__
        elif op.op == "call_method":  # torch.xxx / torch.Tensor.xxx
            real_op = getattr(torch.Tensor, op.target, None)
        elif isinstance(op.target, str):  # named modules
            real_op = named_modules[op.target].__class__
        else:  # builtin object
            real_op = op.target
        real_ops.append(real_op)
    return real_ops


def get_operators(
    ops: List[torch.fx.Node],
    m: torch.fx.GraphModule,
    named_modules: Dict[str, torch.fx.GraphModule],
):
    real_ops = []
    for op in ops:
        if op.op in ["placeholder", "output"]:  # input / output
            real_op = None
        elif op.op == "get_attr":  # parameter or constants
            real_op = recursive_getattr(m, op.target)
        elif op.op == "call_method":  # torch.xxx / torch.Tensor.xxx
            real_op = getattr(torch.Tensor, op.target, None)
        elif isinstance(op.target, str):  # named modules
            real_op = named_modules[op.target]
        else:  # builtin object
            real_op = op.target
        real_ops.append(real_op)
    return real_ops
