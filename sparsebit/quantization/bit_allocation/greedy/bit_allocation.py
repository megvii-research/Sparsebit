from sparsebit.quantization.modules import QConv2d, QLinear, MatMul


def bit_allocation(qmodel, bit_allocated):
    for node in qmodel.model.graph.nodes:
        if node.target in bit_allocated.keys():
            module = getattr(qmodel.model, node.target)
            if isinstance(module, (QConv2d, QLinear)):
                wbit = bit_allocated[node.target]["w"]
                module.weight_quantizer.set_bit(wbit)
                module.weight_quantizer.scale = (
                    module.weight_quantizer._broadcast_qparams(
                        module.weight_quantizer.observer.scales[wbit]
                    )
                )
                module.weight_quantizer.zero_point = (
                    module.weight_quantizer._broadcast_qparams(
                        module.weight_quantizer.observer.zero_points[wbit]
                    )
                )
                fbit = bit_allocated[node.target]["f"]
                module.input_quantizer.set_bit(fbit)
                module.input_quantizer.scale = (
                    module.input_quantizer._broadcast_qparams(
                        module.input_quantizer.observer.scales[fbit]
                    )
                )
                module.input_quantizer.zero_point = (
                    module.input_quantizer._broadcast_qparams(
                        module.input_quantizer.observer.zero_points[fbit]
                    )
                )
            elif isinstance(module, MatMul):
                input_0 = getattr(qmodel.model, node.all_input_nodes[0].target)
                input0_bit = bit_allocated[node.target]["f0"]
                input_0.input_quantizer.set_bit(input0_bit)
                input_0.input_quantizer.scale = (
                    input_0.input_quantizer._broadcast_qparams(
                        input_0.input_quantizer.observer.scales[input0_bit]
                    )
                )
                input_0.input_quantizer.zero_point = (
                    input_0.input_quantizer._broadcast_qparams(
                        input_0.input_quantizer.observer.zero_points[input0_bit]
                    )
                )
                input_1 = getattr(qmodel.model, node.all_input_nodes[1].target)
                input1_bit = bit_allocated[node.target]["f1"]
                input_1.input_quantizer.set_bit(input1_bit)
                input_1.input_quantizer.scale = (
                    input_1.input_quantizer._broadcast_qparams(
                        input_1.input_quantizer.observer.scales[input1_bit]
                    )
                )
                input_1.input_quantizer.zero_point = (
                    input_1.input_quantizer._broadcast_qparams(
                        input_1.input_quantizer.observer.zero_points[input1_bit]
                    )
                )
