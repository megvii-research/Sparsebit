import torch
from sparsebit.quantization.quant_model import QuantModel
from sparsebit.quantization.quant_config import _C as default_config
from examples.quantization_aware_training.cifar10.basecase.model import resnet20


def build_config(changes_list):
    new_config = default_config.clone()
    new_config.defrost()
    new_config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    new_config.merge_from_list(changes_list)
    new_config.freeze()
    return new_config


def test_residual_block():
    # the format of list([k1, v1, k2, v2, ...]), ensure config[::2] is key and config[1::2] is value
    model_config = [
        ("BACKEND", "virtual"),
        ("W.QSCHEME", "per-channel-symmetric"),
        ("W.QUANTIZER.TYPE", "uniform"),
        ("W.QUANTIZER.BIT", 4),
        ("A.QSCHEME", "per-tensor-affine"),
        ("A.QUANTIZER.TYPE", "uniform"),
        ("A.QUANTIZER.BIT", 4),
        ("A.QADD.ENABLE_QUANT", True),
    ]
    model_config = [j for i in model_config for j in i]

    model = resnet20(num_classes=10)
    config = build_config(model_config)
    qmodel = QuantModel(model, config)

    data = torch.randn(1, 3, 32, 32)
    model.eval()
    qmodel.eval()
    out1 = model(data)
    out2 = qmodel(data)
    qmodel.export_onnx(data, "temp.onnx", extra_info=True)
