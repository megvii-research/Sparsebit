import torch
import importlib

from .lists import fuse_list as default_fuse_list
from sparsebit.quantization.tools import fx_symbolic_trace
from sparsebit.quantization.converters.prune import PruneGraph


def fuse_operations(model: torch.fx.GraphModule, config, custom_fuse_list=None):
    model = fx_symbolic_trace(model)
    model = PruneGraph().apply(model)
    fuse_list = custom_fuse_list if custom_fuse_list else default_fuse_list
    for simplify_task in fuse_list:
        if getattr(config, simplify_task.upper(), True):
            simplify_module = importlib.import_module(
                ".{}".format(simplify_task), package=__package__
            )
            if getattr(simplify_module, "ReplacePatterns", None):
                simplify_classes = simplify_module.ReplacePatterns
                for simplify_class in simplify_classes:
                    simplify_class.apply(model)
            else:
                simplify_func = simplify_module.ReplacePattern
                simplify_func().apply(model)
            model = PruneGraph().apply(model)
    return model
