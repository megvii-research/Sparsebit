import torch
from sparsebit.quantization.modules import QConv2d, QLinear, MatMul


def calc_flops_and_limitations(qmodel, target_w_bit, target_a_bit):
    bops_limitation = 0
    bops_limitation_for_search = 0
    memory_limitation = 0
    for node in qmodel.model.graph.nodes:
        if node.op in ["placeholder", "output"]:
            continue
        module = getattr(qmodel.model, node.target)
        if (
            isinstance(module, (QConv2d, QLinear))
            and getattr(module, "input_quantizer", None)
            and getattr(module, "weight_quantizer", None)
            and not module.input_quantizer.fake_fused
        ):
            module.flops = module.weight.numel()
            if isinstance(module, QConv2d):
                module.flops *= module.output_shape[-1] * module.output_shape[-2]
            elif (
                isinstance(module, QLinear)
                and module.input_quantizer.observer.qdesc._ch_axis == 2  # NLC
            ):
                module.flops *= module.output_shape[1]
            if (
                node.target in qmodel.cfg.A.SPECIFIC[0].keys()
                and "QUANTIZER.BIT" in qmodel.cfg.A.SPECIFIC[0][node.target]
                and node.target in qmodel.cfg.W.SPECIFIC[0].keys()
                and "QUANTIZER.BIT" in qmodel.cfg.W.SPECIFIC[0][node.target]
            ):
                bops_limitation += (
                    module.flops
                    * module.input_quantizer.bit
                    * module.weight_quantizer.bit
                )
                memory_limitation += module.weight.numel()  # Byte
            else:
                bops_limitation += module.flops * target_w_bit * target_a_bit
                bops_limitation_for_search += module.flops * target_w_bit * target_a_bit
                memory_limitation += module.weight.numel() * target_w_bit / 8  # Byte
        elif isinstance(module, MatMul) and getattr(
            module, "input_quantizer_generated", None
        ):
            input0_shape = getattr(
                qmodel.model, node.all_input_nodes[0].target
            ).output_shape
            input1_shape = getattr(
                qmodel.model, node.all_input_nodes[1].target
            ).output_shape
            module.flops = (
                torch.prod(torch.tensor(input0_shape[1:])) * input1_shape[-1]
            ).item()

            bops = module.flops * (target_a_bit**2)
            bops_limitation_for_search += bops
            bops_limitation += bops

    return (
        bops_limitation,
        bops_limitation_for_search,
        memory_limitation,
    )


def calc_final_bops_and_memory(qmodel):
    allocated_bops = 0
    allocated_memory = 0
    for node in qmodel.model.graph.nodes:
        if node.op in ["placeholder", "output"]:
            continue
        module = getattr(qmodel.model, node.target)
        if (
            isinstance(module, (QConv2d, QLinear))
            and getattr(module, "input_quantizer", None)
            and getattr(module, "weight_quantizer", None)
            and not module.input_quantizer.fake_fused
        ):
            allocated_bops += (
                module.flops * module.weight_quantizer.bit * module.input_quantizer.bit
            )
            allocated_memory += (
                module.weight.numel() * module.weight_quantizer.bit / 8
            )  # Byte
            print("module name:", node.target)
            print("input bit:", module.input_quantizer.bit)
            print("weight bit:", module.weight_quantizer.bit)
            print()
        elif isinstance(module, MatMul) and getattr(
            module, "input_quantizer_generated", None
        ):
            input0_bit = getattr(
                qmodel.model, node.all_input_nodes[0].target
            ).input_quantizer.bit
            input1_bit = getattr(
                qmodel.model, node.all_input_nodes[1].target
            ).input_quantizer.bit
            allocated_bops += module.flops * input0_bit * input1_bit
            print("module name:", node.target)
            print("input0 bit:", input0_bit)
            print("input1 bit:", input1_bit)
            print()

    return allocated_bops, allocated_memory
