# include "fake_quant_tensor.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quant_pertensor_forward", QuantizePerTensorForward, "A function to fake quant a tensor in pertensor");
    m.def("quant_pertensor_backward", QuantizePerTensorBackward, "A backward function of fake quant in pertensor");
    m.def("quant_perchannel_forward", QuantizePerChannelForward, "A function to fake quant a tensor in perchannel");
    m.def("quant_perchannel_backward", QuantizePerChannelBackward, "A backward function of fake quant in perchannel");
}
