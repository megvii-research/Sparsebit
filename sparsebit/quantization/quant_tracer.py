import torch
from torch.fx import GraphModule, Tracer
from typing import Dict, Any, List, Callable, Tuple, Optional, Set
from fnmatch import fnmatch


class QTracer(Tracer):
    def __init__(self, skipped_module_names: List[str]):
        super().__init__()
        self.skipped_module_names = skipped_module_names

    def _probe(self, module_name, patterns):
        for p in patterns:
            if fnmatch(module_name, p):
                return True
        return False

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return (
            m.__module__.startswith("torch.nn")
            and not isinstance(m, torch.nn.Sequential)
        ) or self._probe(module_qualified_name, self.skipped_module_names)
