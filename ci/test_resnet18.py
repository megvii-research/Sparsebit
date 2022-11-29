import torch
from torchvision.models import resnet18
from sparsebit.quantization.quant_model import QuantModel
from sparsebit.quantization.quant_config import _C as default_config


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
    ]
    model_config = [j for i in model_config for j in i]

    model = resnet18(pretrained=True)
    config = build_config(model_config)
    qmodel = QuantModel(model, config)

    data = torch.randn(1, 3, 224, 224)
    model.eval()
    qmodel.eval()
    out1 = model(data)
    out2 = qmodel(data)
    torch.testing.assert_allclose(out1, out2, atol=1e-4, rtol=1e-4)
