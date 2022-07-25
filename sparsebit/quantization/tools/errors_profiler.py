from typing import Callable
from functools import partial
import torch
import torch.fx as fx
import torch.nn.functional as F


class SharedData(object):
    """用于管理中间计算结果和保存对分结果。"""

    def __init__(self):
        self.outputs = {}
        self.edges = {}
        self.output_degrees = {}
        self.diffs = {}

    def add_node(self, name: str, inputs: list):
        self.output_degrees[name] = 0
        self.edges[name] = [i for i in inputs if i is not None]
        for inp in self.edges[name]:
            if inp not in self.output_degrees:
                self.output_degrees[inp] = 1
            else:
                self.output_degrees[inp] += 1

    def finish_node(self, name):
        for inp in self.edges[name]:
            self.output_degrees[inp] -= 1
            if self.output_degrees[inp] == 0:
                del self.output_degrees[inp]
                del self.outputs[inp]
        if self.output_degrees[name] == 0:
            del self.output_degrees[name]
            del self.outputs[name]

    def set_value(self, name: str, value):
        self.outputs[name] = value

    def get_value(self, name: str):
        if name not in self.outputs:
            return None
        return self.outputs[name]

    def save_diff(self, name: str, diff: torch.Tensor):
        self.diffs[name] = diff.detach()

    def get_output_diff(self):
        output = self.diffs
        self.diffs = {}
        assert len(self.outputs) == 0
        assert len(self.output_degrees) == 0
        return output


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
        self.storage = SharedData()

    def apply(
        self, data: torch.Tensor, checker: Callable = F.mse_loss, is_async: bool = True
    ):
        if is_async:
            self._quantization_error_async(data=data, checker=checker)
        else:
            self._quantization_error_sync(data=data, checker=checker)
        return self.storage.get_output_diff()

    def _quantization_error_async(self, data: torch.Tensor, checker: Callable):
        """用提供的样例输入和对分方法对比量化前后的分数。

        使用异步对分策略。即： 计算"网络中只有该层被量化"时的误差。

        Args:
            data (torch.Tensor): 一个样例输入。
            checker (Callable):
                一个metric,要求接受同一层layer在量化前后的两个tensor输出,产生一个自定义误差值。
        """
        named_modules = dict(self.model.named_modules())
        handles = []

        for name, module in named_modules.items():
            if getattr(module, "input_quantizer", None):
                handles.append(module.register_forward_pre_hook(hook=_forward_pre_hook))
                handles.append(
                    module.register_forward_hook(
                        hook=partial(
                            _forward_hook,
                            name=name,
                            input_names=None,
                            storage=self.storage,
                            checker=checker,
                            check_diff=True,
                            is_async=True,
                        )
                    )
                )

        self.model.forward(data)
        for handle in handles:
            handle.remove()

    def _quantization_error_sync(self, data: torch.Tensor, checker: Callable):
        """用提供的样例输入和对分方法对比量化前后的分数。

        使用同步对分策略。即： 计算"网络中所有层都被量化"时的误差。

        Args:
            data (torch.Tensor): 一个样例输入。
            checker (Callable):
                一个metric,要求接受同一层layer在量化前后的两个tensor输出,产生一个自定义误差值。
        """
        fx_graph = self.model.graph
        named_modules = dict(self.model.named_modules())
        handles = []

        # assume fx_graph.node is topological-sorted
        for node in fx_graph.nodes:
            if node.op in ["placeholder", "output"]:  # skip IO empty node
                continue
            node_name = node.target
            module = named_modules[node_name]

            # 输入module名称，None表示不是module，此时没有量化，可以直接使用float输入值作为量化输入值。
            input_node_names = []
            for input_node in node.args:
                if not isinstance(input_node, torch.fx.node.Node) or input_node.op in [
                    "placeholder",
                    "output",
                ]:
                    input_node_names.append(None)
                else:
                    input_node_names.append(input_node.target)
            self.storage.add_node(node_name, input_node_names)

            forward_hook_func = partial(
                _forward_hook,
                name=node_name,
                input_names=input_node_names,
                storage=self.storage,
                checker=checker,
                is_async=False,
            )
            if node.op == "call_module" and getattr(module, "input_quantizer", None):
                handles.append(module.register_forward_pre_hook(_forward_pre_hook))
                handles.append(
                    module.register_forward_hook(
                        partial(forward_hook_func, check_diff=True)
                    )
                )
            else:
                handles.append(
                    module.register_forward_hook(
                        partial(forward_hook_func, check_diff=False)
                    )
                )

        self.model.forward(data)
        for handle in handles:
            handle.remove()


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
    name: str,
    input_names: str,
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
        pred = module.forward(*f_in)
    else:
        q_in = [
            storage.get_value(i)
            if i is not None
            else _detach(f_in[idx])  # 如果输入层名称是None，直接使用float输入
            for idx, i in enumerate(input_names)
        ]
        # 手动forward可以避免循环调用_forward_hook
        pred = module.forward(*q_in)

    # 设置本module为非量化
    if getattr(module, "input_quantizer", None):
        module.input_quantizer.disable_quant()
    if getattr(module, "weight_quantizer", None):
        module.weight_quantizer.disable_quant()

    # 对分
    pred = _detach(pred)
    gt = _detach(f_out)
    storage.save_diff(name, checker(pred, gt))

    if not is_async:
        storage.set_value(name, pred)
        # (同步对分优化)删除接下来不再使用的中间结果
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


def _detach(x):
    if isinstance(x, torch.Tensor):
        return x.detach()
    if isinstance(x, (list, tuple)):
        return tuple([_detach(i) for i in x])
    return x
