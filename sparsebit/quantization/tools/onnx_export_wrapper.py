from sparsebit.quantization.quantizers import Quantizer


class ExtraInfoContextManager:
    def __init__(self, model):
        self.model = model

    def enable_extra_info(self):
        for module in self.model.modules():
            if isinstance(module, Quantizer):
                module.enable_extra_info()

    def disable_extra_info(self):
        for module in self.model.modules():
            if isinstance(module, Quantizer):
                module.disable_extra_info()

    def __enter__(self):
        self.enable_extra_info()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.disable_extra_info()


class ONNXExportContextManager:
    def __init__(self, model):
        self.model = model

    def enable_export_onnx(self):
        for module in self.model.modules():
            if isinstance(module, Quantizer):
                module.enable_export_onnx()
                # FIXME: if quantizer bit!=8, extra_info must be enabled
                if module.bit != 8 and not module.extra_info:
                    assert (
                        False
                    ), "8bit is supported by default. \
                        You must set extra_info=True when export a model with {}bit".format(
                        module.bit
                    )

    def disable_export_onnx(self):
        for module in self.model.modules():
            if isinstance(module, Quantizer):
                module.disable_export_onnx()

    def __enter__(self):
        self.enable_export_onnx()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.disable_export_onnx()


def enable_extra_info_export(model):
    """
    Usage:
        with enable_extra_info_export(model):
            torch.onnx.export(model, ...)
    """
    return ExtraInfoContextManager(model)


def enable_onnx_export(model):
    """
    Usage:
        with enable_onnx_export(model):
            torch.onnx.export(model, ...)
    """
    return ONNXExportContextManager(model)
