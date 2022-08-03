from typing import Callable
from functools import partial
import torch
import torch.fx as fx
import torch.nn.functional as F

from .tensor_wrapper import to_detach
from .graph_wrapper import GraphVisitor, SharedData


class QuantizationErrorProfiler(object):
    """用提供的样例输入和对分方法对比量化前后的分数。

    提供异步/同步两种对分策略。其中

    - ``异步`` 表示衡量每一层在"网络中只有该层被量化"时的误差。

    - ``同步`` 表示衡量每一层在"网络中所有层都被量化"时的误差。

    调用apply()，输出一个 ``{name: error_value}`` ,表示量化前后每一层对应误差

    Args:
        data (torch.Tensor): 一个样例输入。
        checker (Callable):
            一个metric,要求接受同一层layer在量化前后的两个tensor输出,产生一个自定义误差值。
        is_async (bool): 是否异步运行
    """

    def __init__(self, model: fx.GraphModule):
        self.model = model

    def apply(
        self, data: torch.Tensor, checker: Callable = F.mse_loss, is_async: bool = True
    ):
        if is_async:
            return self._quantization_error_async(data=data, checker=checker)
        else:
            return self._quantization_error_sync(data=data, checker=checker)

    def _quantization_error_async(self, data: torch.Tensor, checker: Callable):
        """用提供的样例输入和对分方法对比量化前后的分数。

        使用异步对分策略。即： 计算"网络中只有该层被量化"时的误差。

        Args:
            data (torch.Tensor): 一个样例输入。
            checker (Callable):
                一个metric,要求接受同一层layer在量化前后的两个tensor输出,产生一个自定义误差值。
        """

        forward_pre_hook = _forward_pre_hook
        forward_hook = partial(_forward_hook, checker=checker, is_async=True)

        def hook_wrapper(
            node,
            module,
            storage: SharedData,
        ):
            handles = []
            handles.append(module.register_forward_pre_hook(hook=forward_pre_hook))
            handles.append(
                module.register_forward_hook(
                    hook=partial(
                        forward_hook,
                        node=node,
                        storage=storage,
                        check_diff=getattr(module, "input_quantizer", None)
                        and module.input_quantizer.is_enable,
                    )
                )
            )
            return handles

        builder = GraphVisitor(self.model, hook_wrapper)

        self.model.forward(data)

        return builder.storage.extract_value("diff")

    def _quantization_error_sync(self, data: torch.Tensor, checker: Callable):
        """用提供的样例输入和对分方法对比量化前后的分数。

        使用同步对分策略。即： 计算"网络中所有层都被量化"时的误差。

        Args:
            data (torch.Tensor): 一个样例输入。
            checker (Callable):
                一个metric,要求接受同一层layer在量化前后的两个tensor输出,产生一个自定义误差值。
        """

        forward_pre_hook = _forward_pre_hook
        forward_hook = partial(_forward_hook, checker=checker, is_async=False)

        def hook_wrapper(
            node,
            module,
            storage: SharedData,
        ):
            handles = []
            if (
                node.op == "call_module"
                and getattr(module, "input_quantizer", None)
                and module.input_quantizer.is_enable
            ):
                handles.append(module.register_forward_pre_hook(forward_pre_hook))
                handles.append(
                    module.register_forward_hook(
                        partial(
                            forward_hook,
                            node=node,
                            storage=storage,
                            check_diff=True,
                        )
                    )
                )
            else:
                handles.append(
                    module.register_forward_hook(
                        partial(
                            forward_hook,
                            node=node,
                            storage=storage,
                            check_diff=False,
                        )
                    )
                )
            return handles

        builder = GraphVisitor(self.model, hook_wrapper)

        self.model.forward(data)

        return builder.storage.extract_value("diff")


def _forward_pre_hook(module, f_in):  # before forward()
    if getattr(module, "input_quantizer", None) and module.input_quantizer.is_enable:
        setattr(module, "reverse_input_quant_flag", True)
        module.input_quantizer.disable_quant()
    else:
        setattr(module, "reverse_input_quant_flag", False)

    if getattr(module, "weight_quantizer", None) and module.weight_quantizer.is_enable:
        setattr(module, "reverse_weight_quant_flag", True)
        module.weight_quantizer.disable_quant()
    else:
        setattr(module, "reverse_weight_quant_flag", False)


def _forward_hook(
    module,
    f_in,
    f_out,
    node: torch.fx.Node,
    storage: SharedData,
    checker: Callable,
    check_diff: bool,
    is_async: bool,
):  # after forward()
    # 设置本module为量化
    if getattr(module, "input_quantizer", None):
        module.input_quantizer.enable_quant()
    if getattr(module, "weight_quantizer", None):
        module.weight_quantizer.enable_quant()

    # 计算量化的结果pred
    if is_async:
        if check_diff:
            pred = to_detach(module.forward(*f_in))
    else:
        q_in = to_detach(storage.extract_node_args(node.args, f_in, batch=None))
        # 手动forward可以避免循环调用_forward_hook
        pred = module.forward(*q_in)

    # 设置本module为非量化
    if getattr(module, "input_quantizer", None):
        module.input_quantizer.disable_quant()
    if getattr(module, "weight_quantizer", None):
        module.weight_quantizer.disable_quant()

    # 对分
    name = node.target
    if check_diff:
        pred = to_detach(pred)
        gt = to_detach(f_out)
        storage.set_value(name, "diff", checker(pred, gt))
    if not is_async:
        storage.set_output(name, pred)

    storage.finish_node(name)

    # 还原module的量化配置
    if hasattr(module, "reverse_input_quant_flag"):
        if module.reverse_input_quant_flag:
            module.input_quantizer.enable_quant()
        delattr(module, "reverse_input_quant_flag")
    if hasattr(module, "reverse_weight_quant_flag"):
        if module.reverse_weight_quant_flag:
            module.weight_quantizer.enable_quant()
        delattr(module, "reverse_weight_quant_flag")
