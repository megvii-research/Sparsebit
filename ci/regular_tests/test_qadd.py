from sparsebit.quantization.modules.unary import QIdentity
import torch
import torch.nn as nn
from sparsebit.quantization.quant_model import QuantModel
from sparsebit.quantization.quant_config import _C as default_config
from sparsebit.quantization.modules.math import QAdd


class ConvAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 3)
        self.conv2 = nn.Conv2d(3, 3, 3)

    def forward(self, x):
        x_left = self.conv1(x)
        x_right = self.conv2(x)
        out = torch.add(x_left, x_right)
        return out


def build_config(changes_list):
    new_config = default_config.clone()
    new_config.defrost()
    new_config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    new_config.merge_from_list(changes_list)
    new_config.freeze()
    return new_config


def test_deit_tiny():
    # the format of list([k1, v1, k2, v2, ...]), ensure config[::2] is key and config[1::2] is value
    model_config = [
        ("BACKEND", "tensorrt"),
        ("SCHEDULE.FUSE_BN", True),
        ("W.QSCHEME", "per-channel-symmetric"),
        ("W.QUANTIZER.TYPE", "uniform"),
        ("W.QUANTIZER.BIT", 8),
        ("W.OBSERVER.TYPE", "MINMAX"),
        ("A.QSCHEME", "per-tensor-symmetric"),
        ("A.QUANTIZER.TYPE", "uniform"),
        ("A.QUANTIZER.BIT", 8),
        ("A.OBSERVER.TYPE", "MINMAX"),
        ("A.OBSERVER.LAYOUT", "NCHW"),
        ("A.QADD.ENABLE_QUANT", True),
    ]
    model_config = [j for i in model_config for j in i]

    model = ConvAdd()
    config = build_config(model_config)
    qmodel = QuantModel(model, config)

    data = torch.randn(1, 3, 4, 4)
    model.eval()
    qmodel.eval()
    out1 = model(data)
    out2 = qmodel(data)
    for node in qmodel.model.graph.nodes:
        module = getattr(qmodel.model, node.target, None)
        if module is not None:
            if isinstance(module, QAdd):
                counter = 0
                for prev_node in node.all_input_nodes:
                    input_module = getattr(qmodel.model, prev_node.target, None)
                    if isinstance(input_module, QIdentity):
                        counter += 1
                assert counter > 0, "Qadd quantizers not build successfully!"
    torch.testing.assert_allclose(out1, out2, atol=1e-4, rtol=1e-4)
