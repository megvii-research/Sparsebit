import torch
import torch.nn as nn
from functools import partial
from sparsebit.quantization.modules import QConv2d, QLinear, MatMul


def get_perturbations(qmodel, data, label):
    qmodel.set_quant(False, False)
    param_dict = {}

    def matmul_hook(module, data_input, data_output, name):
        param_dict[name] = data_input

    hook_handler = []
    for node in qmodel.model.graph.nodes:
        if node.op in ["placeholder", "output"]:
            continue
        module = getattr(qmodel.model, node.target)
        if (
            isinstance(module, MatMul)
            and getattr(module, "input_quantizer_generated", None)
        ):
            if (
                node.target in qmodel.cfg.A.SPECIFIC[0].keys()
                and "QUANTIZER.BIT" in qmodel.cfg.A.SPECIFIC[0][node.target]
            ):
                continue
            handler = module.register_forward_hook(
                partial(matmul_hook, name=node.target)
            )
            hook_handler.append(handler)

    data.requires_grad_() 
    output = qmodel(data.cuda())
    loss = nn.CrossEntropyLoss()(output, label.cuda())
    calib_acc = (output.max(1)[1] == label.to(output.device)).sum().item()/output.shape[0]
    print("calib_acc:", str(calib_acc*100)+"%")
    # for handler in hook_handler:
    #     handler.remove()

    sensitivities = []
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

            grads= torch.autograd.grad(loss, module.weight, create_graph=True)
            weight_sensitivity = calc_block_sensitivity(grads[0].reshape(-1), module.weight).cpu().item()
            print(
                "weight_sensitivity:",
                str(weight_sensitivity),
                # "feature_sensitivity:",
                # str(feature_sensitivity),
            )
            sensitivities.append(weight_sensitivity)
            # feature_sensitivity = calc_block_sensitivity(grads[1].reshape(-1), param_dict[node.target][1]).cpu().item()
            

            module.set_quant(True, False)
            delta_w = {}
            # delta_f = {}
            for w_bit in module.weight_quantizer.observer.scales.keys():
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
                with torch.no_grad():
                    weight_q = module.weight_quantizer(module.weight)
                delta_w[w_bit] = (weight_q - module.weight).reshape(-1)

            # for f_bit in module.input_quantizer.observer.scales.keys():
            #     module.input_quantizer.set_bit(f_bit)
            #     module.input_quantizer.scale = (
            #         module.input_quantizer._broadcast_qparams(
            #             module.input_quantizer.observer.scales[f_bit]
            #         )
            #     )
            #     module.input_quantizer.zero_point = (
            #         module.input_quantizer._broadcast_qparams(
            #             module.input_quantizer.observer.zero_points[f_bit]
            #         )
            #     )
            #     with torch.no_grad():
            #         feature_q = module.input_quantizer(param_dict[node.target][1])
            #     delta_f[f_bit] = (feature_q - param_dict[node.target][1]).reshape(-1)

            perturbations_conv_linear[node.target] = {}
            for w_bit, d_w in delta_w.items():
                perturbation = (abs(weight_sensitivity)*torch.norm(d_w, p=2, dim=0) ** 2).item()
                perturbations_conv_linear[node.target][w_bit] = perturbation
                print(
                    "    " + str(w_bit) + "w" + str(w_bit) + "f",
                    "Perturbation:",
                    str(perturbation),
                )
            module.set_quant(False, False)

        elif (
            isinstance(module, MatMul)
            and getattr(module, "input_quantizer_generated", None)
        ):
            if (
                node.target in qmodel.cfg.A.SPECIFIC[0].keys()
                and "QUANTIZER.BIT" in qmodel.cfg.A.SPECIFIC[0][node.target]
            ):
                continue
            print("Layer name:", node.target)
            print("FLOPs:", module.flops)

            grads= torch.autograd.grad(loss, param_dict[node.target], create_graph=True)
            input0_sensitivity = calc_block_sensitivity(grads[0].reshape(-1), param_dict[node.target][0]).cpu().item()
            input1_sensitivity = calc_block_sensitivity(grads[1].reshape(-1), param_dict[node.target][1]).cpu().item()
            print(
                "input0_sensitivity:",
                str(input0_sensitivity),
                "input1_sensitivity:",
                str(input1_sensitivity),
            )

            input0 = getattr(qmodel.model, node.all_input_nodes[0].target)
            input1 = getattr(qmodel.model, node.all_input_nodes[1].target)
            input0.set_quant(False, True)
            input1.set_quant(False, True)
            delta_f0 = {}
            delta_f1 = {}
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
                with torch.no_grad():
                    feature0_q = input0.input_quantizer(param_dict[node.target][0])
                delta_f0[input0_bit] = (feature0_q - param_dict[node.target][0]).reshape(-1)

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
                    feature1_q = input1.input_quantizer(param_dict[node.target][1])
                delta_f1[input1_bit] = (feature1_q - param_dict[node.target][1]).reshape(-1)

            perturbations_matmul[node.target] = {}
            for f0_bit, d_f0 in delta_f0.items():
                perturbations_matmul[node.target][f0_bit] = {}
                for f1_bit, d_f1 in delta_f1.items():
                    perturbation = ((abs(input0_sensitivity)*torch.norm(d_f0, p=2, dim=0) ** 2)+(abs(input1_sensitivity)*torch.norm(d_f1, p=2, dim=0) ** 2)).item()
                    perturbations_matmul[node.target][f0_bit][f1_bit] = perturbation
                    print(
                        "    ",
                        str(f0_bit) + "/" + str(f1_bit) + "f",
                        "Perturbation:",
                        str(perturbation),
                    )
            input0.set_quant(False, False)
            input1.set_quant(False, False)
            
    print(sensitivities)
    return perturbations_conv_linear, perturbations_matmul


def rademacher(shape, dtype=torch.float32):
    """Sample from Rademacher distribution."""
    rand = ((torch.rand(shape) < 0.5)) * 2 - 1
    return rand.to(dtype).cuda()


def calc_block_sensitivity(
    block_grads, derived_params, eigen_type="avg", sensitivity_calc_iter_num=50
):
    if eigen_type == "max":
        v = torch.randn(block_grads.shape).cuda()
        v = v / torch.norm(v, p=2)
        for _ in range(sensitivity_calc_iter_num):
            gv_grads = (
                torch.autograd.grad(block_grads, derived_params, grad_outputs=v, retain_graph=True)[0]
                .reshape(-1)
            )
            gv_grads_norm = torch.norm(gv_grads, p=2)
            v = gv_grads / gv_grads_norm
            max_eigen_value = gv_grads_norm
        return max_eigen_value

    elif eigen_type == "avg":
        trace = 0
        v = rademacher(block_grads.shape)
        for _ in range(sensitivity_calc_iter_num):
            Hv = (
                torch.autograd.grad(block_grads, derived_params, grad_outputs=v, retain_graph=True)[0]
                .reshape(-1)
            )
            vHv = torch.matmul(v, Hv)
            trace += vHv / sensitivity_calc_iter_num
            v = rademacher(block_grads.shape)
        avg_eigen_value = trace / Hv.shape[0]
        return avg_eigen_value