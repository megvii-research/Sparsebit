from sparsebit.quantization.modules import QConv2d, QLinear

def calc_flops_and_limitations(model, target_w_bit, target_a_bit):
    bops_limitation = 0
    bops_limitation_for_feature_search = 0
    memory_limitation = 0
    for node in model.graph.nodes:
        if node.op in ["placeholder", "output"]:
            continue
        module = getattr(model, node.target)
        if (
            isinstance(module, (QConv2d, QLinear))
            and getattr(module, "input_quantizer", None)
            and getattr(module, "weight_quantizer", None)
            and not module.input_quantizer.fake_fused
        ):
            module.flops = module.weight.numel()
            if isinstance(module, QConv2d):
                module.flops *= module.output_hw[0]*module.output_hw[1]
            bops = module.flops*target_w_bit*target_a_bit
            bops_limitation_for_feature_search += module.flops*8*target_a_bit
            bops_limitation += bops
            memory_limitation += module.weight.numel()*target_w_bit/8 #Byte

    return bops_limitation, bops_limitation_for_feature_search, memory_limitation

