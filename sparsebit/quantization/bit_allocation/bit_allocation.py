from sparsebit.quantization.modules import QConv2d, QLinear, MatMul


def feature_bit_allocation(qmodel, bit_allocated):
    for node in qmodel.model.graph.nodes:
        if node.target in bit_allocated.keys():
            bit = bit_allocated[node.target]
            module = getattr(qmodel.model, node.target)
            if isinstance(module, (QConv2d, QLinear)):
                module.input_quantizer.set_bit(bit)
                module.input_quantizer.scale = (
                    module.input_quantizer._broadcast_qparams(
                        module.input_quantizer.observer.scales[bit]
                    )
                )
                module.input_quantizer.zero_point = (
                    module.input_quantizer._broadcast_qparams(
                        module.input_quantizer.observer.zero_points[bit]
                    )
                )
            elif isinstance(module, MatMul):
                input_0 = getattr(qmodel.model, node.all_input_nodes[0].target)
                input_1 = getattr(qmodel.model, node.all_input_nodes[1].target)
                input_0.input_quantizer.set_bit(bit)
                input_0.input_quantizer.scale = (
                    input_0.input_quantizer._broadcast_qparams(
                        input_0.input_quantizer.observer.scales[bit]
                    )
                )
                input_0.input_quantizer.zero_point = (
                    input_0.input_quantizer._broadcast_qparams(
                        input_0.input_quantizer.observer.zero_points[bit]
                    )
                )
                input_1.input_quantizer.set_bit(bit)
                input_1.input_quantizer.scale = (
                    input_1.input_quantizer._broadcast_qparams(
                        input_1.input_quantizer.observer.scales[bit]
                    )
                )
                input_1.input_quantizer.zero_point = (
                    input_1.input_quantizer._broadcast_qparams(
                        input_1.input_quantizer.observer.zero_points[bit]
                    )
                )


def weight_bit_allocation(qmodel, bit_allocated):
    for node in qmodel.model.graph.nodes:
        if node.target in bit_allocated.keys():
            bit = bit_allocated[node.target]
            module = getattr(qmodel.model, node.target)
            module.weight_quantizer.set_bit(bit)
            module.weight_quantizer.scale = module.weight_quantizer._broadcast_qparams(
                module.weight_quantizer.observer.scales[bit]
            )
            module.weight_quantizer.zero_point = (
                module.weight_quantizer._broadcast_qparams(
                    module.weight_quantizer.observer.zero_points[bit]
                )
            )
