from sparsebit.quantization.modules import QuantOpr


def feature_bit_allocation(qmodel, bit_allocated):
    for node in qmodel.model.graph.nodes:
        if node.target in bit_allocated.keys():
            bit = bit_allocated[node.target]
            module = getattr(qmodel.model, node.target)
            module.input_quantizer.set_bit(bit)
            module.input_quantizer.scale = module.input_quantizer._broadcast_qparams(
                module.input_quantizer.observer.scales[bit]
            )
            module.input_quantizer.zero_point = (
                module.input_quantizer._broadcast_qparams(
                    module.input_quantizer.observer.zero_points[bit]
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
