from sparsebit.quantization.modules import QConv2d, QLinear

def mse(pred, target, p=2.0):
    return (pred - target).abs().pow(p).mean()


def get_perturbations(qmodel, data):
    qmodel.set_quant(False, False)
    float_output = qmodel(data.cuda())
    weight_perturbation = {}
    feature_perturbation = {}
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
            print("Perturbation of layer name:", node.target)
            print("    Feature:")
            feature_perturbation[node.target] = {}
            for bit in module.input_quantizer.observer.scales.keys():
                module.input_quantizer.set_bit(bit)
                module.input_quantizer.scale = module.input_quantizer._broadcast_qparams(module.input_quantizer.observer.scales[bit])
                module.input_quantizer.zero_point = module.input_quantizer._broadcast_qparams(module.input_quantizer.observer.zero_points[bit])
                module.set_quant(False, True)
                quant_output = qmodel(data.cuda())
                perturbation = mse(float_output, quant_output)
                module.set_quant(False, False)
                print("        Bit:", str(bit), "Perturbation:", str(perturbation.item()))
                feature_perturbation[node.target][bit] = perturbation.item()

            print("    Weight:")
            weight_perturbation[node.target] = {}
            for bit in module.weight_quantizer.observer.scales.keys():
                module.weight_quantizer.set_bit(bit)
                module.weight_quantizer.scale = module.weight_quantizer._broadcast_qparams(module.weight_quantizer.observer.scales[bit])
                module.weight_quantizer.zero_point = module.weight_quantizer._broadcast_qparams(module.weight_quantizer.observer.zero_points[bit])
                module.set_quant(True, False)
                quant_output = qmodel(data.cuda())
                perturbation = mse(float_output, quant_output)
                module.set_quant(False, False)
                print("        Bit:", str(bit), "Perturbation:", str(perturbation.item()))
                weight_perturbation[node.target][bit] = perturbation.item()

    return feature_perturbation, weight_perturbation
