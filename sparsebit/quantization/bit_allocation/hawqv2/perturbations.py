import torch
from functools import partial
from sparsebit.quantization.modules import QConv2d, QLinear, MatMul


def mse(pred, target, p=2.0):
    return (pred - target).abs().pow(p).mean()


def calc_sensitivity(loss, param):
    return


def get_perturbations(qmodel, data):
    qmodel.set_quant(False, False)
    with torch.no_grad():
        float_output = qmodel(data.cuda())

    input_features = {}
    weights = {}

    def feature_hook(module, data_input, data_output, name):
        input_features[name] = data_input

    hook_handler = []
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
            isinstance(module, MatMul)
            and getattr(module, "input_quantizer_generated", None)
        ):
            handler = module.register_forward_hook(
                partial(feature_hook, name=node.target)
            )
            hook_handler.append(handler)

    data.requires_grad = True
    qmodel.set_quant(True, False)
    quant_output = qmodel(data.cuda())
    loss = mse(float_output, quant_output)
    for handler in hook_handler:
        handler.remove()

    feature_perturbation = get_feature_perturbation(qmodel, loss, input_features)
    weight_perturbation = get_weight_perturbation(qmodel, loss, weights)

    return feature_perturbation, weight_perturbation


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
            gv = torch.matmul(block_grads, v)
            gv_grads = (
                torch.autograd.grad(gv, derived_params, retain_graph=True)[0]
                .contiguous()
                .view(-1)
            )
            gv_grads_norm = torch.norm(gv_grads, p=2)
            v = gv_grads / gv_grads_norm
            max_eigen_value = gv_grads_norm
        return max_eigen_value

    elif eigen_type == "avg":
        trace = 0
        v = rademacher(block_grads.shape)
        for _ in range(sensitivity_calc_iter_num):
            gv = torch.matmul(block_grads, v)
            gv_grads = (
                torch.autograd.grad(gv, derived_params, retain_graph=True)[0]
                .contiguous()
                .view(-1)
            )
            vHv = torch.matmul(v, gv_grads)
            trace += vHv / sensitivity_calc_iter_num
            v = rademacher(block_grads.shape)
        avg_eigen_value = trace / gv_grads.shape[0]
        return avg_eigen_value


def get_feature_perturbation(qmodel, loss, input_features):
    perturbation = {}
    for layer_name, feature in input_features.items():
        perturbation[layer_name] = {}
        grads = (
            torch.autograd.grad(loss, feature, create_graph=True)[0]
            .contiguous()
            .view(-1)
        )

        sensitivity = calc_block_sensitivity(grads, feature).cpu().item()
        print(sensitivity)

        module = getattr(qmodel.model, layer_name)
        perturbation[layer_name] = {}
        module.set_quant(False, True)
        for bit in module.input_quantizer.observer.scales.keys():
            module.input_quantizer.set_bit(bit)
            module.input_quantizer.scale = module.input_quantizer._broadcast_qparams(
                module.input_quantizer.observer.scales[bit]
            )
            module.input_quantizer.zero_point = (
                module.input_quantizer._broadcast_qparams(
                    module.input_quantizer.observer.zero_points[bit]
                )
            )
            feature_q = module.input_quantizer(feature)
            delta_f = (feature_q - feature).reshape(-1)
            perturbation[layer_name][bit] = sensitivity * (
                torch.norm(delta_f, p=2, dim=0) ** 2
            )
        module.set_quant(False, False)
    return perturbation


def get_weight_perturbation(qmodel, loss, weights):
    perturbation = {}
    for layer_name, weight in weights.items():
        perturbation[layer_name] = {}
        grads = (
            torch.autograd.grad(loss, weight, create_graph=True)[0]
            .contiguous()
            .view(-1)
        )

        sensitivity = calc_block_sensitivity(grads, weight).cpu().item()
        print(sensitivity)

        module = getattr(qmodel.model, layer_name)
        perturbation[layer_name] = {}
        module.set_quant(True, False)
        for bit in module.input_quantizer.observer.scales.keys():
            module.input_quantizer.set_bit(bit)
            module.input_quantizer.scale = module.input_quantizer._broadcast_qparams(
                module.input_quantizer.observer.scales[bit]
            )
            module.input_quantizer.zero_point = (
                module.input_quantizer._broadcast_qparams(
                    module.input_quantizer.observer.zero_points[bit]
                )
            )
            weight_q = module.weight_quantizer(weight)
            delta_w = (weight_q - weight).reshape(-1)
            perturbation[layer_name][bit] = sensitivity * (
                torch.norm(delta_w, p=2, dim=0) ** 2
            )
        module.set_quant(False, False)
    return perturbation
