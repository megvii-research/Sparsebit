import copy
import torch
import torch.nn as nn
import torchvision.models as models

from sparsebit.quantization.quant_model import QuantModel
from sparsebit.quantization.quant_config import _C as default_config


def build_qconfig(changes_list):
    new_config = default_config.clone()
    new_config.defrost()
    new_config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    new_config.merge_from_list(changes_list)
    new_config.freeze()
    return new_config


def normal_calibration(model, inputs, config):
    qmodel = QuantModel(model, config)
    qmodel.prepare_calibration()
    qmodel.eval()
    # run calibration
    with torch.no_grad():
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        output = qmodel(inputs)
    qmodel.calc_qparams()
    print("finish normal calibration")


def asym_calibration_wquant(model, inputs, config):
    qmodel = QuantModel(model, config)
    qmodel.prepare_calibration()
    qmodel.eval()
    # run calibration
    with torch.no_grad():
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        output = qmodel(inputs)
    qmodel.calc_qparams(asym=True, w_quant=True, a_quant=False)
    print("finish asym calibration with w_quant")


def asym_calibration_aquant(model, inputs, config):
    qmodel = QuantModel(model, config)
    qmodel.prepare_calibration()
    qmodel.eval()
    # run calibration
    with torch.no_grad():
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        output = qmodel(inputs)
    qmodel.calc_qparams(asym=True, w_quant=False, a_quant=True)
    print("finish asym calibration with a_quant")


def asym_calibration_waquant(model, inputs, config):
    qmodel = QuantModel(model, config)
    qmodel.prepare_calibration()
    qmodel.eval()
    # run calibration
    with torch.no_grad():
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        output = qmodel(inputs)
    qmodel.calc_qparams(asym=True, w_quant=True, a_quant=True)
    print("finish asym calibration with wa_quant")


def test_calibration():
    qconfig = [
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
    qconfig = [j for i in qconfig for j in i]
    qconfig = build_qconfig(qconfig)
    model = models.__dict__["resnet18"](pretrained=False)
    rand_inputs = torch.randn(4, 3, 224, 224)
    normal_calibration(model, rand_inputs, qconfig)
    asym_calibration_wquant(model, rand_inputs, qconfig)
    asym_calibration_aquant(model, rand_inputs, qconfig)
    asym_calibration_waquant(model, rand_inputs, qconfig)


if __name__ == "__main__":
    test_calibration()
