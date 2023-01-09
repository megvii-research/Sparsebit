from sparsebit.quantization.modules import QConv2d, QLinear, MatMul


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
        if (
            node.target in qmodel.cfg.A.SPECIFIC[0].keys()
            and "QUANTIZER.BIT" in qmodel.cfg.A.SPECIFIC[0][node.target]
            and node.target in qmodel.cfg.W.SPECIFIC[0].keys()
            and "QUANTIZER.BIT" in qmodel.cfg.W.SPECIFIC[0][node.target]
        ):
            continue
        module = getattr(qmodel.model, node.target)
        if (
            isinstance(module, (QConv2d, QLinear))
            and getattr(module, "input_quantizer", None)
            and getattr(module, "weight_quantizer", None)
            and not module.input_quantizer.fake_fused
        ):
            print("Layer name:", node.target)
            print("FLOPs:", module.flops)
            print("    Feature:")
            feature_perturbation[node.target] = {}
            for bit in module.input_quantizer.observer.scales.keys():
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
                module.set_quant(False, True)
                quant_output = qmodel(data.cuda())
                perturbation = mse(float_output, quant_output)
                module.set_quant(False, False)
                print(
                    "        Bit:", str(bit), "Perturbation:", str(perturbation.item())
                )
                feature_perturbation[node.target][bit] = perturbation.item()

            print("    Weight:")
            weight_perturbation[node.target] = {}
            for bit in module.weight_quantizer.observer.scales.keys():
                module.weight_quantizer.set_bit(bit)
                module.weight_quantizer.scale = (
                    module.weight_quantizer._broadcast_qparams(
                        module.weight_quantizer.observer.scales[bit]
                    )
                )
                module.weight_quantizer.zero_point = (
                    module.weight_quantizer._broadcast_qparams(
                        module.weight_quantizer.observer.zero_points[bit]
                    )
                )
                module.set_quant(True, False)
                quant_output = qmodel(data.cuda())
                perturbation = mse(float_output, quant_output)
                module.set_quant(False, False)
                print(
                    "        Bit:", str(bit), "Perturbation:", str(perturbation.item())
                )
                weight_perturbation[node.target][bit] = perturbation.item()
        elif isinstance(module, (MatMul)) and getattr(
            module, "input_quantizer_generated", None
        ):
            print("Layer name:", node.target)
            print("FLOPs:", module.flops)
            print("    Feature:")
            input_0 = getattr(qmodel.model, node.all_input_nodes[0].target)
            input_1 = getattr(qmodel.model, node.all_input_nodes[1].target)
            feature_perturbation[node.target] = {}
            for bit in input_0.input_quantizer.observer.scales.keys():
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
                input_0.set_quant(False, True)
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
                input_1.set_quant(False, True)
                quant_output = qmodel(data.cuda())
                perturbation = mse(float_output, quant_output)
                input_0.set_quant(False, False)
                input_1.set_quant(False, False)
                print(
                    "        Bit:", str(bit), "Perturbation:", str(perturbation.item())
                )
                feature_perturbation[node.target][bit] = perturbation.item()

    return feature_perturbation, weight_perturbation
