import torch


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.cpu()
    if isinstance(x, tuple):
        return tuple([to_cpu(i) for i in x])
    if isinstance(x, list):
        return [to_cpu(i) for i in x]
    if isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    return x


def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, tuple):
        return tuple([to_device(i, device) for i in x])
    if isinstance(x, list):
        return [to_device(i, device) for i in x]
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


def to_detach(x):
    if isinstance(x, torch.Tensor):
        return x.detach()
    if isinstance(x, tuple):
        return tuple([to_detach(i) for i in x])
    if isinstance(x, list):
        return [to_detach(i) for i in x]
    if isinstance(x, dict):
        return {k: to_detach(v) for k, v in x.items()}
    return x
