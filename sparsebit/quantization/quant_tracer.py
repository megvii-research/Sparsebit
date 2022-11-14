import torch
from torch.fx import GraphModule, Tracer
from typing import Dict, Any, List, Callable, Tuple, Optional, Set


class QTracer(Tracer):
    def __init__(
            self,
            skipped_module_names: List[str]):
            #skipped_module_classes: List[Callable]):
        super().__init__()
        self.skipped_module_names = skipped_module_names
        #self.skipped_module_classes = skipped_module_classes

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str) -> bool:
        return (m.__module__.startswith("torch.nn") and
                not isinstance(m, torch.nn.Sequential)) or \
            module_qualified_name in self.skipped_module_names
            #or type(m) in self.skipped_module_classes
