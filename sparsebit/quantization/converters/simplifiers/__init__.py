import torch
import importlib

from .lists import simplify_list
from sparsebit.quantization.tools import fx_symbolic_trace
from sparsebit.quantization.converters.prune import PruneGraph


def simplify(model: torch.fx.GraphModule):
    model = fx_symbolic_trace(model)
    model = PruneGraph().apply(model)
    for simplify_task in simplify_list:
        simplify_module = importlib.import_module(
            ".{}".format(simplify_task), package=__package__
        ).ReplacePattern
        simplify_module().apply(model)
        model = PruneGraph().apply(model)
    return model
