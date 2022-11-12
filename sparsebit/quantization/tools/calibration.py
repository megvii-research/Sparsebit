import torch
from functools import partial

from sparsebit.quantization.modules import QuantOpr
from .graph_wrapper import GraphVisitor, fx_symbolic_trace
from .tensor_wrapper import to_cpu, to_device, to_detach


class CalibrationRunner(object):
    def __init__(self, model):
        self.model = fx_symbolic_trace(model)

    def prepare_calibration(self):
        input_names_cache = set(
            i.target for i in self.model.graph.nodes if i.op == "placeholder"
        )

        # tile outputs in input nodes
        def _forward_hook(module, x_in, x_out, node, storage, record_names):
            def flatten(x):
                if isinstance(x, (list, tuple)):
                    out = []
                    for i in x:
                        out.extend(flatten(i))
                    return tuple(out)
                return (x,)

            pos = 0
            flatten_x_in = flatten(x_in)
            flatten_args = flatten(node.args)
            for _x_in, _args in zip(flatten_x_in, flatten_args):
                if isinstance(_args, torch.fx.Node) and _args.target in record_names:
                    input_name = _args.target
                    datas = storage.get_output(input_name)
                    if datas is None:
                        datas = []
                    datas.append(to_cpu(to_detach(x_in[pos])))
                    storage.set_output(input_name, datas)

        def hook_wrapper(node, module, storage):
            hooks = []
            input_names = [
                inp_node.target
                for inp_node in node.all_input_nodes
                if inp_node.target in input_names_cache
            ]
            if len(input_names) > 0:
                hooks.append(
                    module.register_forward_hook(
                        hook=partial(
                            _forward_hook,
                            node=node,
                            storage=storage,
                            record_names=input_names,
                        )
                    )
                )
                for input_name in input_names:
                    input_names_cache.remove(input_name)

            return hooks

        self.builder = GraphVisitor(self.model, hook_wrapper)

    def feature_layerwise_calibration(self, device):
        # manual forward once to calculate calibration
        assert hasattr(self, "builder"), "run self.prepare_calibration first!"
        batch_num = None
        for node in self.model.graph.nodes:
            if node.op in ["placeholder", "output"]:
                if batch_num is None:
                    batch_num = len(self.builder.storage.get_output(node.target))
                continue

            assert batch_num is not None

            module = getattr(self.model, node.target)
            if isinstance(module, QuantOpr) and getattr(
                module, "input_quantizer", None
            ):
                for inp_node in node.all_input_nodes:
                    inp_tensors = self.builder.storage.get_output(inp_node.target)
                    for inp_tensor in inp_tensors:
                        module.input_quantizer.update_observer(inp_tensor)
                module.input_quantizer.calc_qparams()
                module.input_quantizer.observer.data_cache.reset()

            with torch.no_grad():
                outputs = []
                for batch_idx in range(batch_num):
                    if node.op == "get_attr":  # is constant value
                        outputs.append(to_cpu(module.data))
                        continue
                    args = self.builder.storage.extract_node_args(
                        node.args, batch=batch_idx
                    )
                    args = to_device(args, device)
                    # more time for less cuda memory occupation
                    outputs.append(to_cpu(module(*args, **node.kwargs)))
            self.builder.storage.set_output(node.target, outputs)
            self.builder.storage.finish_node(node.target)

    def weight_calibration(self):
        for n, m in self.model.named_modules():
            if isinstance(m, QuantOpr) and getattr(m, "weight_quantizer", None):
                m.weight_quantizer.update_observer(m.weight)
                m.weight_quantizer.calc_qparams()
