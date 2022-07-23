import torch.fx as fx


def fx_symbolic_trace(model):
    if not getattr(model, "graph", None):
        model = fx.symbolic_trace(model)
    return model
