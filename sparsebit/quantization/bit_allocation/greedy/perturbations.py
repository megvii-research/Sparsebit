import torch
from sparsebit.quantization.modules import QConv2d, QLinear, MatMul


def mse(pred, target, p=2.0):
    return (pred - target).abs().pow(p).mean()


def get_perturbations(qmodel, data):
    qmodel.set_quant(False, False)
    with torch.no_grad():
        float_output = qmodel(data.cuda())
    perturbations_conv_linear = {}
    perturbations_matmul = {}
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
            if (
                node.target in qmodel.cfg.A.SPECIFIC[0].keys()
                and "QUANTIZER.BIT" in qmodel.cfg.A.SPECIFIC[0][node.target]
                and node.target in qmodel.cfg.W.SPECIFIC[0].keys()
                and "QUANTIZER.BIT" in qmodel.cfg.W.SPECIFIC[0][node.target]
            ):
                continue
            print("Layer name:", node.target)
            print("FLOPs:", module.flops)
            perturbations_conv_linear[node.target] = {}
            module.set_quant(True, True)
            for w_bit in module.weight_quantizer.observer.scales.keys():
                perturbations_conv_linear[node.target][w_bit] = {}
                module.weight_quantizer.set_bit(w_bit)
                module.weight_quantizer.scale = (
                    module.weight_quantizer._broadcast_qparams(
                        module.weight_quantizer.observer.scales[w_bit]
                    )
                )
                module.weight_quantizer.zero_point = (
                    module.weight_quantizer._broadcast_qparams(
                        module.weight_quantizer.observer.zero_points[w_bit]
                    )
                )
                for f_bit in module.input_quantizer.observer.scales.keys():
                    module.input_quantizer.set_bit(f_bit)
                    module.input_quantizer.scale = (
                        module.input_quantizer._broadcast_qparams(
                            module.input_quantizer.observer.scales[f_bit]
                        )
                    )
                    module.input_quantizer.zero_point = (
                        module.input_quantizer._broadcast_qparams(
                            module.input_quantizer.observer.zero_points[f_bit]
                        )
                    )

                    with torch.no_grad():
                        quant_output = qmodel(data.cuda())
                    perturbation = mse(float_output, quant_output).item()
                    print(
                        "    " + str(w_bit) + "w" + str(f_bit) + "f",
                        "Perturbation:",
                        str(perturbation),
                    )
                    perturbations_conv_linear[node.target][w_bit][f_bit] = perturbation
            module.set_quant(False, False)

        elif isinstance(module, (MatMul)) and getattr(
            module, "input_quantizer_generated", None
        ):
            if (
                node.target in qmodel.cfg.A.SPECIFIC[0].keys()
                and "QUANTIZER.BIT" in qmodel.cfg.A.SPECIFIC[0][node.target]
            ):
                continue
            print("Layer name:", node.target)
            print("FLOPs:", module.flops)
            input0 = getattr(qmodel.model, node.all_input_nodes[0].target)
            input1 = getattr(qmodel.model, node.all_input_nodes[1].target)
            input0.set_quant(False, True)
            input1.set_quant(False, True)
            perturbations_matmul[node.target] = {}
            for input0_bit in input0.input_quantizer.observer.scales.keys():
                perturbations_matmul[node.target][input0_bit] = {}
                input0.input_quantizer.set_bit(input0_bit)
                input0.input_quantizer.scale = (
                    input0.input_quantizer._broadcast_qparams(
                        input0.input_quantizer.observer.scales[input0_bit]
                    )
                )
                input0.input_quantizer.zero_point = (
                    input0.input_quantizer._broadcast_qparams(
                        input0.input_quantizer.observer.zero_points[input0_bit]
                    )
                )

                for input1_bit in input1.input_quantizer.observer.scales.keys():
                    input1.input_quantizer.set_bit(input1_bit)
                    input1.input_quantizer.scale = (
                        input1.input_quantizer._broadcast_qparams(
                            input1.input_quantizer.observer.scales[input1_bit]
                        )
                    )
                    input1.input_quantizer.zero_point = (
                        input1.input_quantizer._broadcast_qparams(
                            input1.input_quantizer.observer.zero_points[input1_bit]
                        )
                    )

                    with torch.no_grad():
                        quant_output = qmodel(data.cuda())
                    perturbation = mse(float_output, quant_output).item()
                    print(
                        "    ",
                        str(input0_bit) + "/" + str(input1_bit) + "f",
                        "Perturbation:",
                        str(perturbation),
                    )
                    perturbations_matmul[node.target][input0_bit][
                        input1_bit
                    ] = perturbation
            input0.set_quant(False, False)
            input1.set_quant(False, False)

    return perturbations_conv_linear, perturbations_matmul
