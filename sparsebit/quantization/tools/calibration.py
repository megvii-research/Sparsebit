import torch
from functools import partial

from sparsebit.quantization.modules import QuantOpr
from sparsebit.quantization.tools import SharedData


def get_topo(model):
    calib_cache = SharedData()
    for node in model.graph.nodes:
        if node.op in ["output"]:  # skip output empty node
            continue
        node_name = node.name

        # 输入module名称，None表示不是module，此时没有量化，可以直接使用float输入值作为量化输入值。
        input_node_names = []
        for input_node in node.args:
            if not isinstance(input_node, torch.fx.node.Node) or input_node.op in [
                "output",
            ]:
                input_node_names.append(None)
            else:
                input_node_names.append(input_node.name)
        calib_cache.add_node(node_name, input_node_names)
    return calib_cache


def register_hook(model, calib_cache):
    def _forward_hook(module, x_in, x_out, name):
        prev_data = calib_cache.get_value(name)
        if prev_data is None:
            prev_data = []
        assert isinstance(
            prev_data, list
        ), "calibration data must be cached in list format!"
        prev_data.append(x_in[0].cpu())
        calib_cache.set_value(name, prev_data)

    calibration_handles = []
    for node in model.graph.nodes:
        if node.op == "placeholder":
            for n in node.users:
                module = getattr(model, n.target)
                quant_node = n
                while not (isinstance(module, QuantOpr) and module.input_quantizer):
                    quant_node = n.next
                    module = getattr(model, quant_node.target)

                assert (
                    len(module._forward_hooks) == 0
                ), "only support one input in calibration now"

                h = module.register_forward_hook(
                    hook=partial(_forward_hook, name=quant_node.prev.name)
                )
                calibration_handles.append(h)
    return calibration_handles


def feature_layerwise_calibration(model, calib_cache, device):
    for node in model.graph.nodes:
        if (
            node.op in ["placeholder", "output"]
            or calib_cache.get_value(node.prev.name) is None
        ):
            continue
        module = getattr(model, node.target)
        if isinstance(module, QuantOpr) and module.input_quantizer:
            for x in calib_cache.get_value(node.prev.name):
                module.input_quantizer.update_observer(x)
            module.input_quantizer.calc_qparams()
            module.input_quantizer.observer.reset_data_cache()
        inputs = zip(
            *[
                calib_cache.get_value(node.all_input_nodes[i].name)
                for i in range(len(node.all_input_nodes))
            ]
        )
        with torch.no_grad():
            outputs = []
            for batch in inputs:
                batch = [input.to(device) for input in batch]
                outputs.append(module(*batch).cpu())
        calib_cache.set_value(node.name, outputs)
        calib_cache.finish_node(node.name)

    assert len(calib_cache.outputs) == 0
    assert len(calib_cache.output_degrees) == 0
    del calib_cache
