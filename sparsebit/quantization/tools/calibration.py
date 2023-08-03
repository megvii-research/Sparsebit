import copy
import torch
from functools import partial

from sparsebit.quantization.modules import QuantOpr
from sparsebit.quantization.quantizers.adaround import reconstruct_qlayer
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

            flatten_x_in = flatten(x_in)
            flatten_args = flatten(node.args)
            for pos, (_x_in, _args) in enumerate(zip(flatten_x_in, flatten_args)):
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

    def layerwise_calibration(self, device, asym=False, w_quant=False, a_quant=False):
        """
        asym: enable calibration with all preceding layers quantized
        w_quant: quant the weights in all preceding layers
        a_quant: quant the inputs in all preceding layers
        """
        # manual forward once to calculate calibration
        assert hasattr(self, "builder"), "run self.prepare_calibration first!"
        # remove hook from module before calibration
        for handle in self.builder.handles:
            handle.remove()
        if asym:
            self.builder.qstorage = copy.deepcopy(self.builder.storage)
        # run calibration qopr-by-qopr
        batch_num = None
        for node in self.model.graph.nodes:
            if node.op in ["placeholder", "output"]:
                if batch_num is None:
                    batch_num = len(self.builder.storage.get_output(node.target))
                continue
            assert batch_num is not None
            self.run_feature_calibration(node, asym)
            # forward float output
            float_outputs = self.module_forward(batch_num, node, device)
            self.builder.storage.set_output(node.target, float_outputs)
            self.run_weight_calibration(node, asym, a_quant=a_quant)
            # foward quant output
            if asym:
                quant_outputs = self.module_forward(
                    batch_num, node, device, asym, w_quant, a_quant
                )
                self.builder.qstorage.set_output(node.target, quant_outputs)
                self.builder.qstorage.finish_node(node.target)
            # pop the outputs of nodes whose out-degree=0
            self.builder.storage.finish_node(node.target)

    def run_feature_calibration(self, node, asym=False):
        module = getattr(self.model, node.target)
        if (
            isinstance(module, QuantOpr)
            and getattr(module, "input_quantizer", None)
            and not module.input_quantizer.fake_fused
        ):
            for inp_node in node.all_input_nodes:
                inp_tensors = self.builder.storage.get_output(inp_node.target)
                for inp_tensor in inp_tensors:
                    if isinstance(inp_tensor, torch.Tensor):
                        module.input_quantizer.update_observer(inp_tensor)
            module.input_quantizer.calc_qparams()
            module.input_quantizer.observer.data_cache.reset()

    def run_weight_calibration(self, node, asym=False, a_quant=False):
        module = getattr(self.model, node.target)
        if isinstance(module, QuantOpr) and getattr(module, "weight_quantizer", None):
            module.weight_quantizer.update_observer(module.weight)
            module.weight_quantizer.calc_qparams()
            if module.weight_quantizer.TYPE.lower() == "adaround":
                assert (
                    len(node.all_input_nodes) == 1
                ), "AdaRound not supports the oprs which has more than one inputs"
                _storage = self.builder.qstorage if asym else self.builder.storage
                inp_tensors = _storage.get_output(node.all_input_nodes[0].target)
                out_tensors = self.builder.storage.get_output(node.target)
                print("Reconstruct {}".format(node.target))
                reconstruct_qlayer(
                    module,
                    torch.cat(inp_tensors, dim=0),
                    torch.cat(out_tensors, dim=0),
                    a_quant=a_quant,
                )

    def module_forward(
        self, batch_num, node, device, asym=False, w_quant=False, a_quant=False
    ):
        module = getattr(self.model, node.target)
        if node.op == "call_module":
            module.eval()
        if isinstance(module, QuantOpr) and asym:
            module.set_quant(w_quant, a_quant)
        with torch.no_grad():
            outputs = []
            for batch_idx in range(batch_num):
                if node.op == "get_attr":  # is constant value
                    outputs.append(to_cpu(module.data))
                    continue
                storage = self.builder.qstorage if asym else self.builder.storage
                args = storage.extract_node_args(node.args, batch=batch_idx)
                kwargs = storage.extract_node_kwargs(node.kwargs, batch=batch_idx)
                args = to_device(args, device)
                kwargs = to_device(kwargs, device)
                # more time for less cuda memory occupation
                outputs.append(to_cpu(module(*args, **kwargs)))
        if isinstance(module, QuantOpr):
            module.set_quant(w_quant=False, a_quant=False)
            module.output_shape = outputs[0].shape
        return outputs
