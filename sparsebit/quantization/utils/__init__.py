import torch
from sparsebit.quantization.quant_tracer import QTracer


def fx_symbolic_trace(model):
    if not getattr(model, "graph", None):
        model = torch.fx.symbolic_trace(model)
    return model
