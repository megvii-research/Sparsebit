import torch
import importlib

from .lists import lists as default_lists
from sparsebit.quantization.tools import fx_symbolic_trace
from sparsebit.quantization.converters.prune import PruneGraph


def fuse_operations(model: torch.fx.GraphModule, config, custom_lists=None):
    model = fx_symbolic_trace(model)
    model = PruneGraph().apply(model)
    cur_list = custom_lists if custom_lists else default_lists
    for task in cur_list:
        if getattr(config, task.upper(), True):
            module = importlib.import_module(".{}".format(task), package=__package__)
            if getattr(module, "ReplacePatterns", None):
                classes = module.ReplacePatterns
                for cls in classes:
                    cls.apply(model)
            else:
                func = module.ReplacePattern
                func().apply(model)
            model = PruneGraph().apply(model)
    return model
