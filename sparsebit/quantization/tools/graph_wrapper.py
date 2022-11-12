from typing import Callable
import torch
import torch.fx as fx


def fx_symbolic_trace(model):
    if not getattr(model, "graph", None):
        model = fx.symbolic_trace(model)
    return model


class SharedData(object):
    """用于管理中间结果。"""

    def __init__(self):
        self.values = {}
        self.edges = {}
        self.output_degrees = {}
        self.outputs = {}

    def __del__(self):
        assert len(self.output_degrees) == 0
        assert len(self.outputs) == 0
        assert len(self.values) == 0, "use extract_value to get out all the results"

    def add_node(self, name: str, inputs: list):
        self.output_degrees[name] = 0
        self.edges[name] = [i for i in inputs if i is not None]
        for inp in self.edges[name]:
            if inp not in self.output_degrees:
                self.output_degrees[inp] = 1
            else:
                self.output_degrees[inp] += 1

    def finish_node(self, name):
        for inp in self.edges[name]:
            self.output_degrees[inp] -= 1
            if self.output_degrees[inp] == 0:
                del self.output_degrees[inp]
                self.outputs.pop(inp, None)
        if self.output_degrees[name] == 0:
            del self.output_degrees[name]
            self.outputs.pop(name, None)

    def set_output(self, name: str, value):
        self.outputs[name] = value

    def set_value(self, name: str, value_name: str, value):
        assert value_name != "output"
        if value_name not in self.values:
            self.values[value_name] = {}
        self.values[value_name][name] = value

    def get_output(self, name: str):
        return self.outputs.get(name, None)

    def get_value(self, name: str, value_name: str):
        assert value_name != "output"
        if value_name not in self.values or name not in self.values[value_name]:
            return None
        return self.values[value_name][name]

    def extract_node_args(self, args, real_input: tuple = None, batch: int = None):
        # usage: extract_node_args(node.args, x_in) in forward_hook
        if isinstance(args, fx.Node):
            input = self.get_output(args.target)
            if not isinstance(input, torch.Tensor) and not input:
                input = real_input
            return input[batch] if batch is not None else input

        if isinstance(args, (list, tuple)):
            inputs = [
                self.extract_node_args(i, j, batch=batch)
                for i, j in zip(args, real_input if real_input else [None] * len(args))
            ]
            if isinstance(args, tuple):
                inputs = tuple(inputs)
            return inputs

        return args

    def extract_value(self, value_name: str):
        return self.values.pop(value_name, {})


class GraphVisitor(object):
    def __init__(self, model: fx.GraphModule, hook_wrapper: Callable):
        self.storage = SharedData()
        self.build(model, hook_wrapper)

    def __del__(self):
        for handle in self.handles:
            handle.remove()

    def build(self, model: fx.GraphModule, hook_wrapper: Callable):
        fx_graph = model.graph
        named_modules = dict(model.named_modules())

        self.handles = []
        # assume fx_graph.node is topological-sorted
        for node in fx_graph.nodes:
            if node.op in ["placeholder", "output"]:  # skip IO empty node
                continue
            if node.op == "get_attr":  # use model.xxx to get constant nn.Parameter
                module = getattr(model, node.target)
            else:
                module = named_modules[node.target]

            input_node_targets = [
                input_node.target for input_node in node.all_input_nodes
            ]
            self.storage.add_node(node.target, input_node_targets)

            ret = hook_wrapper(node=node, module=module, storage=self.storage,)
            if ret is not None:
                self.handles.extend(ret)
