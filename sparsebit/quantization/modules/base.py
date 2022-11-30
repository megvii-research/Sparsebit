import torch
import torch.nn as nn
from sparsebit.quantization.common import get_backend
from sparsebit.quantization.quantizers import build_quantizer
from sparsebit.utils import update_config
from sparsebit.quantization.common import QuantTarget


class QuantOpr(nn.Module):
    """QuantOpr是torch算子的量化版本。
    它提供可配置的 ``input_quantizer`` 和 ``weight_quantizer`` ,
    可根据需要启用。启用后,将转出QDQ格式的onnx模型,便于tensorRT运行。

    Attributes:
        input_quantizer (sparsebit.quantization.quantizers.base.Quantizer):
            输入量化器。
        weight_quantizer (sparsebit.quantization.quantizers.base.Quantizer):
            参数量化器。仅在该算子存在 ``weight`` 属性时可启用。
    """

    def __init__(self):
        super(QuantOpr, self).__init__()
        self.weight = None
        self.input_quantizer = None
        self.weight_quantizer = None

    def forward(self, x_in: torch.Tensor):
        """在考虑量化前提下，描述算子前向传播。

        基类不包含算子的实现。请在子类中添加。
        """
        raise NotImplementedError(
            "no found a forward in {}".format(self.__class__.__name__)
        )

    def build_quantizer(self, config):
        """根据config配置 ``input_quantizer`` 和 ``weight_quantizer`` 。"""
        _backend = get_backend(config.BACKEND)
        if self.weight is not None:
            update_config(config.W, "TARGET", (QuantTarget.WEIGHT,))
            self.weight_quantizer = build_quantizer(cfg=config.W)
            self.weight_quantizer.set_backend(_backend)
        update_config(config.A, "TARGET", (QuantTarget.FEATURE,))
        self.input_quantizer = build_quantizer(cfg=config.A)
        self.input_quantizer.set_backend(_backend)

    def set_quant(self, w_quant: bool = False, a_quant: bool = False):
        """开关本算子的 ``input_quantizer`` 和 ``weight_quantizer`` 。

        .. Note::

            注意 ``input_quantizer`` 和 ``weight_quantizer`` 同时被设置。
            如果只设置其中一个,另一个将被默认设置为关闭。
        """
        if self.weight_quantizer:
            if w_quant and not self.weight_quantizer.fake_fused:
                self.weight_quantizer.enable_quant()
            else:
                self.weight_quantizer.disable_quant()
        if self.input_quantizer:
            if a_quant and not self.input_quantizer.fake_fused:
                self.input_quantizer.enable_quant()
            else:
                self.input_quantizer.disable_quant()

    def __repr__(self):
        info = self._repr_info
        if self.weight_quantizer and self.weight_quantizer.is_enable:
            info += "\n\tweight_quantizer: {}".format(self.weight_quantizer.__repr__())
        if self.input_quantizer and self.input_quantizer.is_enable:
            info += "\n\tinput_quantizer: {}".format(self.input_quantizer.__repr__())

        return info


class MultipleInputsQuantOpr(nn.Module):
    """MultipleInputsQuantOpr是torch算子的多输入量化版本。
    它不会提供 ``input_quantizer`` 和 ``weight_quantizer`` ,
    而是在build_quantizer时对每个输入插入一个独立 ``QIdentity`` 算子，在算子中包含 ``input_quantizer`` 。
    请注意本算子自身不做量化。
    """

    def __init__(self):
        super(MultipleInputsQuantOpr, self).__init__()
        self.input_quantizer_generated = False

    def prepare_input_quantizer(self, node, model):
        from .unary import QIdentity

        if self.input_quantizer_generated:
            return

        input_nodes_cache = list(node.all_input_nodes)
        for idx, input_node in enumerate(input_nodes_cache):
            new_module_name = node.name + "_identity{}".format(idx)
            new_module = QIdentity()
            model.add_module(new_module_name, new_module)
            with model.graph.inserting_before(node):
                identity_node = model.graph.create_node(
                    op="call_module",
                    target=new_module_name,
                    args=(input_node,),
                    kwargs={},
                    name=new_module_name,
                )
            node.replace_input_with(input_node, identity_node)

        self.input_quantizer_generated = True
