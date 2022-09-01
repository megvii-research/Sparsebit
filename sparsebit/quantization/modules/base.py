import torch
import torch.nn as nn
from sparsebit.quantization.common import get_backend
from sparsebit.quantization.quantizers import build_quantizer


class QuantOpr(nn.Module):
    """QuantOpr是torch算子的量化版本。
    它提供可配置的 ``input_quantizer`` 和 ``weight_quantizer`` ,
    可根据需要启用。启用后,将转出QDQ格式的onnx模型,便于tensorRT运行。

    Attributes:
        input_quantizer (sparsebit.quantization.quantizers.base.Quantizer):
            输入量化器。
        weight_quantizer (sparsebit.quantization.quantizers.base.Quantizer):
            参数量化器。仅在该算子存在 ``weight`` 属性时可启用。
        fake_fused (bool):
            一个flag,用于对不需要量化的算子关闭量化。

    .. Warning::

        请不要直接修改 ``fake_fused`` 参数,
        因为这样不会对应修改 ``input_quantizer`` 和 ``weight_quantizer`` 。
        如果需要关闭量化,请调用 ``set_fake_fused()`` 。
    """

    def __init__(self):
        super(QuantOpr, self).__init__()
        self.weight = None
        self.input_quantizer = None
        self.weight_quantizer = None
        self.fake_fused = False  # a flag 用于表示该算子是否被fake_fused, 若fused则不进行量化

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
            self.weight_quantizer = build_quantizer(cfg=config.W)
            self.weight_quantizer.set_backend(_backend)
        self.input_quantizer = build_quantizer(cfg=config.A)
        self.input_quantizer.set_backend(_backend)

    def set_fake_fused(self):
        """关闭 ``input_quantizer`` 和 ``weight_quantizer`` 。
        转换出的QDQ格式onnx在此算子处将没有quantize-dequantize算子。

        仅在不需要量化的地方调用。

        .. Note::

            该过程不可逆。
        """
        self.fake_fused = True
        if self.weight_quantizer:
            self.weight_quantizer.set_fake_fused()
        if self.input_quantizer:
            self.input_quantizer.set_fake_fused()

    def set_quant(self, w_quant: bool = False, a_quant: bool = False):
        """开关本算子的 ``input_quantizer`` 和 ``weight_quantizer`` 。

        .. Note::

            注意 ``input_quantizer`` 和 ``weight_quantizer`` 同时被设置。
            如果只设置其中一个,另一个将被默认设置为关闭。
        """
        if self.weight_quantizer:
            if w_quant and not self.fake_fused:
                self.weight_quantizer.enable_quant()
            else:
                self.weight_quantizer.disable_quant()
        if self.input_quantizer:
            if a_quant and not self.fake_fused:
                self.input_quantizer.enable_quant()
            else:
                self.input_quantizer.disable_quant()

    def __repr__(self):
        info = self._repr_info + "fake_fused: {}".format(self.fake_fused)
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
