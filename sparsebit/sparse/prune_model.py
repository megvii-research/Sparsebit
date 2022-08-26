from atexit import register
import copy
import operator
import importlib
from functools import partial
from fnmatch import fnmatch
from yacs.config import CfgNode as CN
from collections import defaultdict

import torch
import torch.nn as nn
import torch.fx as fx
import torch.nn.functional as F
import onnx

from sparsebit.utils import update_config
from sparsebit.sparse.modules import *
from sparsebit.quantization.converters import simplify  # FIXME

__all__ = ["PruneModel"]


class PruneModel(nn.Module):
    def __init__(self, model: nn.Module, config):
        super().__init__()
        self.model = model
        self.config = config
        self.device = torch.device(config.DEVICE)
        self._run_simplifiers()
        self._convert2prunemodule()
        self._build_pruner()

    def _convert2prunemodule(self):
        """
        将网络中部分node转成对应的prune_module
        """
        named_modules = dict(self.model.named_modules(remove_duplicate=False))
        traced = fx.symbolic_trace(self.model)
        traced.graph.print_tabular()
        pnodes = []  # 用于避免重复遍历
        for n in traced.graph.nodes:
            if not isinstance(n, fx.Node) or n in pnodes:
                continue
            elif n.op == "call_module":
                assert n.target in named_modules, "no found {} in model".format(
                    n.target
                )
                if type(named_modules[n.target]) in PMODULE_MAP:
                    org_module = named_modules[n.target]
                    new_module = PMODULE_MAP[type(org_module)](org_module)
                else:
                    new_module = named_modules[n.target]
            elif n.op in [
                "call_function",
                "call_method",
                "placeholder",
                "get_attr",
                "output",
            ]:
                continue

            with traced.graph.inserting_after(n):
                traced.add_module(n.name, new_module)
                new_node = traced.graph.call_module(n.name, n.args, n.kwargs)
                pnodes.append(new_node)
                n.replace_all_uses_with(
                    new_node
                )  # n的输出全部接到new_node, n成为no user节点(即可删除)

                traced.graph.erase_node(n)
        traced.recompile()
        self.model = fx.GraphModule(traced, traced.graph)

    def _build_pruner(self):
        """
        递归对每个PruneModule建立pruner
        """

        def _probe(module_name: str, specific_modules: dict):
            for k, v in specific_modules.items():
                if fnmatch(module_name, k):
                    return True, v
            return False, None

        def _sub_build(src, module_name):
            sub_cfg = src.clone()
            is_match, specific_config = (
                _probe(module_name, sub_cfg.SPECIFIC[0])
                if src.SPECIFIC
                else (False, None)
            )
            if is_match:
                sub_cfg.merge_from_list(specific_config)
            update_config(sub_cfg, "SPECIFIC", [])
            return sub_cfg

        # build config for every PruneModule
        for n, m in self.model.named_modules():
            if isinstance(m, PruneOpr):
                _config = self.config.clone()  # init
                update_config(_config, "PRUNER", _sub_build(self.config.PRUNER, n))
                m.build_pruner(_config)

    def calc_params(self):
        for node in self.model.graph.nodes:
            if node.op in ["placeholder", "output"]:
                continue
            if node.op == "call_module":
                module = getattr(self.model, node.target, None)
                if isinstance(module, PruneOpr) and getattr(module, "pruner", None):
                    module.calc_mask()

    def _run_simplifiers(self):
        self.model = simplify(self.model)

    def prepare_calibration(self):
        pass

    def forward(self, *args):
        return self.model.forward(*args)

    def export_onnx(
        self,
        dummy_data,
        name,
        input_names=None,
        output_names=None,
        dynamic_axes=None,
        opset_version=13,
        verbose=False,
        extra_info=False,
    ):
        self.eval()

        torch.onnx.export(
            self.model.cpu(),
            dummy_data.cpu(),
            name,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=verbose,
        )
