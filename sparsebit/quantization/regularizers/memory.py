import torch
from sparsebit.quantization.regularizers import Regularizer as BaseRegularizer
from sparsebit.quantization.regularizers import register_regularizer
from sparsebit.quantization.modules import QConv2d, QLinear, MatMul


@register_regularizer
class Regularizer(BaseRegularizer):
    TYPE = "memory"

    def __init__(self,
        config,
        qmodel,
        coeff = 1e6,
    ):
        super(Regularizer, self).__init__(config)
        self.config = config
        self.coeff = coeff
        self.module_dict = {}
        self.memory_limitation = 0
        for node in qmodel.model.graph.nodes:
            if node.op in ["placeholder", "output"]:
                continue
            module = getattr(qmodel.model, node.target)
            if (
                isinstance(module, (QConv2d, QLinear))
                and getattr(module, "weight_quantizer", None)
            ):
                self.module_dict[node.target] = module.weight
                self.memory_limitation += module.weight.numel() * config.W.QUANTIZER.BIT/(2**23)
                self.module_dict[node.target] = {
                    "weight_numel": module.weight.numel(),
                    "qmax": module.weight_quantizer.qmax,
                }

        print("Memory limitation of the model:", str(self.memory_limitation), "MB")

    def __call__(self):
        current_memory = 0
        for n, dict in self.module_dict.items():
            bit = torch.sqrt(2*dict["qmax"]+2)
            current_memory += dict["weight_numel"]*bit/(2**23)
        if current_memory.item()<=self.memory_limitation:
            return 0
        return self.coeff*(current_memory-self.memory_limitation)