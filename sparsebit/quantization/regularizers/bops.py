import torch
from sparsebit.quantization.regularizers import Regularizer as BaseRegularizer
from sparsebit.quantization.regularizers import register_regularizer
from sparsebit.quantization.modules import QConv2d, QLinear, MatMul
from zmq import device


@register_regularizer
class Regularizer(BaseRegularizer):
    TYPE = "bops"

    def __init__(self,
        config,
        qmodel,
        coeff = 1e6,
    ):
        super(Regularizer, self).__init__(config)
        self.config = config
        self.coeff = coeff
        self.module_dict = {}
        self.bops_limitation = 0
        for node in qmodel.model.graph.nodes:
            if node.op in ["placeholder", "output"]:
                continue
            module = getattr(qmodel.model, node.target)
            if (
                isinstance(module, (QConv2d, QLinear))
                and getattr(module, "input_quantizer", None)
                and getattr(module, "weight_quantizer", None)
            ):
                flops = module.weight.numel()
                if isinstance(module, QConv2d):
                    flops *= module.output_shape[-1] * module.output_shape[-2]
                elif (
                    isinstance(module, QLinear)
                    and module.input_quantizer.observer.qdesc._ch_axis == 2  # NLC
                ):
                    flops *= module.output_shape[1]

                self.module_dict[node.target] = {
                    "flops": flops,
                    "is_symmetric1":module.weight_quantizer.qdesc.is_symmetric,
                    "is_symmetric2":module.input_quantizer.qdesc.is_symmetric,
                    "qmax1": module.weight_quantizer.qmax,
                    "qmax2": module.input_quantizer.qmax,
                }
                self.bops_limitation += (
                    flops
                    * config.A.QUANTIZER.BIT
                    * config.W.QUANTIZER.BIT
                )/1e9
            elif isinstance(module, MatMul) and getattr(
                module, "input_quantizer_generated", None
            ):
                input0_quantizer = getattr(qmodel.model, node.all_input_nodes[0].target)
                input1_quantizer = getattr(qmodel.model, node.all_input_nodes[0].target)
                input0_shape = input0_quantizer.output_shape
                input1_shape = input1_quantizer.output_shape
                flops = (
                    torch.prod(torch.tensor(input0_shape[1:])) * input1_shape[-1]
                ).item()
                self.module_dict[node.target] = {
                    "flops": flops,
                    "is_symmetric1":input0_quantizer.qdesc.is_symmetric,
                    "is_symmetric2":input1_quantizer.qdesc.is_symmetric,
                    "qmax1": input0_quantizer.qmax,
                    "qmax2": input1_quantizer.qmax,
                }
                self.bops_limitation += flops * (config.A.QUANTIZER.BIT**2)/1e9

        print("BOPS limitation of the model:", str(self.bops_limitation), "GBOPS")

    def __call__(self):
        current_bops = 0
        for n, dict in self.module_dict.items():
            bit1 = torch.sqrt(2*dict["qmax1"]+2) if dict["is_symmetric1"] else torch.sqrt(dict["qmax1"]+1)
            bit2 = torch.sqrt(2*dict["qmax2"]+2) if dict["is_symmetric2"] else torch.sqrt(dict["qmax2"]+1)
            current_bops += dict["flops"]*bit1*bit2/1e9
        if current_bops.item()<=self.bops_limitation:
            return torch.zeros(1, device=current_bops.device)
        return self.coeff*(current_bops-self.bops_limitation)