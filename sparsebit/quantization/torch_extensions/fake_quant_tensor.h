/*
Code are based on 
https://github.com/openppl-public/ppq/blob/deec01b14a61cdd72627fc0de03882c699723e76/ppq/csrc/cuda/linear.h
Copyright(C) OpenPPL, Apache License 2.0
*/

# include "common.cuh"

Tensor QuantizePerTensorForward(
    const Tensor &data, const Tensor &scale, const Tensor &zero_point,
    const int qmin, const int qmax, const Rounding rounding);


std::vector<Tensor> QuantizePerTensorBackward(
    const Tensor &data, const Tensor &scale, const Tensor &zero_point, 
    const Tensor &grad_y, const int qmin, const int qmax, const Rounding rounding);


Tensor QuantizePerChannelForward(
    const Tensor &data, 
    const Tensor &scale,
    const Tensor &zero_point,
    const int qmin, 
    const int qmax, 
    const int ch_axis, 
    const Rounding rounding);

std::vector<Tensor> QuantizePerChannelBackward(
    const Tensor &data,
    const Tensor &scale, 
    const Tensor &zero_point,
    const Tensor &grad_y,
    const int qmin, 
    const int qmax, 
    const int ch_axis,
    const Rounding rounding);

