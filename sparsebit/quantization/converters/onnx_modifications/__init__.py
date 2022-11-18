import onnx
import importlib

from .lists import onnx_modification_list


def onnx_modifications(model: onnx.ModelProto, modification_list = None, check_model: bool = False):
    if modification_list is None:
        modification_list = onnx_modification_list
    for task in modification_list:
        module = importlib.import_module(".{}".format(task), package=__package__)
        if getattr(module, "ReplacePatterns", None):
            cls = module.ReplacePatterns
            for cur_cls in cls:
                model = cur_cls.apply(model)
        else:
            cur_cls = module.ReplacePattern
            model = cur_cls().apply(model)
    if check_model:
        onnx.checker.check_model(model)
    return model
