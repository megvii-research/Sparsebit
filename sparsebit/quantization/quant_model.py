from atexit import register
from contextlib import contextmanager
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
from sparsebit.quantization.modules import *
from sparsebit.quantization.observers import Observer
from sparsebit.quantization.quantizers import Quantizer
from sparsebit.quantization.tools import QuantizationErrorProfiler
from sparsebit.quantization.converters import simplify, fuse_operations
from sparsebit.quantization.quant_tracer import QTracer


__all__ = ["QuantModel"]


class QuantModel(nn.Module):
    def __init__(self, model: nn.Module, config):
        super().__init__()
        self.cfg = config
        self.device = torch.device(config.DEVICE)
        self.model = self._trace(model)
        self._run_simplifiers()
        self._convert2quantmodule()
        self._build_quantizer()
        self._run_fuse_operations()

    def _convert2quantmodule(self):
        """
        将网络中所有node转成对应的quant_module
        """

        def _get_new_qmodule(target, org_module):
            is_matched = False
            for p in self.cfg.SKIP_TRACE_MODULES:
                if fnmatch(target, p):
                    is_matched = True
            if is_matched:
                new_module = copy.deepcopy(org_module)
            else:
                new_module = QMODULE_MAP[type(org_module)](org_module)
            return new_module

        named_modules = dict(self.model.named_modules(remove_duplicate=False))
        traced = self.model
        modules_viewed = {}
        qnodes = []  # 用于避免重复遍历
        for n in traced.graph.nodes:
            if not isinstance(n, fx.Node) or n in qnodes:
                continue
            elif n.op == "call_module":
                assert n.target in named_modules, "no found {} in model".format(
                    n.target
                )
                org_module = named_modules[n.target]
                if org_module.__module__.startswith("sparsebit.quantization"):
                    qnodes.append(n)
                    continue
                new_module = _get_new_qmodule(n.target, org_module)
            elif n.op == "call_function":
                new_module = QMODULE_MAP[n.target](n, self.cfg)  # node作为module传入获取相关参数
            elif n.op == "call_method":
                if isinstance(n.target, str):
                    target_op = getattr(torch.Tensor, n.target)
                else:
                    raise NotImplementedError
                new_module = QMODULE_MAP[target_op](n, self.cfg)  # node作为module传入获取相关参数
            elif n.op in ["placeholder", "get_attr", "output"]:
                continue
            with traced.graph.inserting_after(n):
                traced.add_module(n.name, new_module)
                new_node = traced.graph.call_module(n.name, n.args, n.kwargs)
                qnodes.append(new_node)
                # n的输出全部接到new_node, n成为no user节点(即可删除)
                n.replace_all_uses_with(new_node)
                traced.graph.erase_node(n)
        traced.recompile()
        self.model = fx.GraphModule(traced, traced.graph)

    def _build_quantizer(self):
        """
        递归对每个QuantModule建立quantizer
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

        # build config for every QuantModule
        original_nodes_cache = list(self.model.graph.nodes)
        for node in original_nodes_cache:
            if node.op == "call_module":
                module = getattr(self.model, node.target)
                if isinstance(module, QuantOpr):
                    _config = self.cfg.clone()  # init
                    update_config(_config, "W", _sub_build(self.cfg.W, node.target))
                    update_config(_config, "A", _sub_build(self.cfg.A, node.target))
                    module.build_quantizer(_config)
                elif (
                    isinstance(module, MultipleInputsQuantOpr)
                    and len(node.all_input_nodes) > 1
                ):
                    module.prepare_input_quantizer(node, self.model)
                    if module.input_quantizer_generated:
                        for input_node in node.all_input_nodes:
                            identity_module = getattr(self.model, input_node.target)
                            _config = self.cfg.clone()  # init
                            update_config(
                                _config, "A", _sub_build(self.cfg.A, node.target)
                            )
                            identity_module.build_quantizer(_config)

    def _trace(self, model):
        skipped_modules = self.cfg.SKIP_TRACE_MODULES
        tracer = QTracer(skipped_modules)
        graph = tracer.trace(model)
        name = (
            model.__class__.__name__
            if isinstance(model, torch.nn.Module)
            else model.__name__
        )
        traced = fx.GraphModule(tracer.root, graph, name)
        return traced

    def _run_simplifiers(self):
        self.model = simplify(self.model)

    def _run_fuse_operations(self):
        if self.cfg.SCHEDULE.BN_TUNING:  # first disable fuse bn
            update_config(self.cfg.SCHEDULE, "FUSE_BN", False)
        self.model = fuse_operations(self.model, self.cfg.SCHEDULE)
        self.model.graph.print_tabular()

    @contextmanager
    def batchnorm_tuning(self):
        """
        We impl a batchnorm tuning algorithm to adjust the stats which will be noisy by quantization.
        Ref:
            batchnorm_tuning: https://arxiv.org/pdf/2006.10518.pdf
        """
        # prepare batchnorm tuning
        self.model.train()
        self.set_quant(w_quant=True, a_quant=True)
        for n, m in self.model.named_modules():
            if isinstance(m, QBatchNorm2d):
                m.module.num_batches_tracked = m.module.num_batches_tracked.zero_()
        yield
        self.model.eval()
        update_config(self.cfg.SCHEDULE, "FUSE_BN", True)
        self.model = fuse_operations(
            self.model, self.cfg.SCHEDULE, custom_fuse_list=["fuse_bn"]
        )
        self.set_quant(w_quant=False, a_quant=False)

    def prepare_calibration(self):
        """
        对与input相接的QuantOpr注册hook, (weight_quantizer不需要)
        """
        from sparsebit.quantization.tools.calibration import CalibrationRunner

        self.eval()
        self.calibration_runner = CalibrationRunner(self.model)
        self.calibration_runner.prepare_calibration()

    def calc_qparams(self):
        assert hasattr(self, "calibration_runner"), "run self.prepare_calibration first"
        self.calibration_runner.feature_layerwise_calibration(self.device)
        self.calibration_runner.weight_calibration()
        del self.calibration_runner

    def init_QAT(self):
        self.calc_qparams()
        self.set_quant(w_quant=True, a_quant=True)
        self.enable_qat = True  # flag, 留备用

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def get_quantization_error(
        self, data: torch.Tensor, checker=F.mse_loss, is_async: bool = True
    ):

        from sparsebit.quantization.tools import QuantizationErrorProfiler

        return QuantizationErrorProfiler(self.model).apply(data, checker, is_async)

    def set_quant(self, w_quant=False, a_quant=False):
        for n, m in self.model.named_modules():
            if isinstance(m, QuantOpr):
                m.set_quant(w_quant, a_quant)

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
        from sparsebit.quantization.tools.onnx_export_wrapper import (
            enable_onnx_export,
            enable_extra_info_export,
        )

        self.eval()
        self.set_quant(w_quant=True, a_quant=True)  # quant must prepared before export

        with enable_onnx_export(self.model, extra_info=extra_info):
            torch.onnx.export(
                self.model.cpu(),
                dummy_data,
                name,
                opset_version=opset_version,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=verbose,
            )
            if extra_info:
                with enable_extra_info_export(self.model):
                    torch.onnx.export(
                        self.model.cpu(),
                        dummy_data.cpu(),
                        name.replace(".onnx", "_external.onnx"),
                        opset_version=opset_version,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=dynamic_axes,
                        verbose=verbose,
                    )
