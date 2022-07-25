/*
Code are based on 
https://github.com/openppl-public/ppq/blob/deec01b14a61cdd72627fc0de03882c699723e76/ppq/csrc/cuda/common.cuh
Copyright(C) OpenPPL, Apache License 2.0
*/

# include <math.h>
# include <cmath>
# include <cuda.h>
# include <cuda_runtime.h>
# include <torch/extension.h>
# include <ATen/cuda/CUDAContext.h>
# pragma once

using at::Tensor;
using Rounding = int;

constexpr int64_t CUDA_NUM_THREADS     = 512;
constexpr int64_t CUDA_TARGET_BLOCKS   = 2560;

constexpr int ROUND_HALF_EVEN          = 0;
constexpr int ROUND_HALF_UP            = 1;
constexpr int ROUND_HALF_DOWN          = 2;

constexpr unsigned int FINAL_MASK      = 0xffffffff;

class ValueTypeException: public std::exception {
public:
    explicit ValueTypeException(const char *m) : message{m} {}
    explicit ValueTypeException(const std::string m) : message{m} {}
    const char *what() const noexcept override { return message.c_str(); }
private:
    std::string message = "";
};

class InvalidValueException: public std::exception {
public:
    explicit InvalidValueException(const char *m) : message{m} {}
    explicit InvalidValueException(const std::string m) : message{m} {}
    const char *what() const noexcept override { return message.c_str(); }
private:
    std::string message = "";
};

__host__ inline
void CheckTensor(const Tensor &tensor, const c10::ScalarType &type, const std::string &name){
    if(at::typeMetaToScalarType(tensor.dtype()) != type){
        throw ValueTypeException(
            std::move("Kernel Failure, Invalid dtype of Input tensor: " + name));
    }
    if(tensor.numel() == 0){
        throw InvalidValueException(
            std::move("Kernel Failure, Tensor is empty: " + name));
    }
}

__host__ inline
int64_t NUM_OF_BLOCK(int64_t elements){
    return std::min((elements + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS, CUDA_TARGET_BLOCKS);
}


__device__ inline
int _round2int(
    const float value,
    const int rounding
){
    switch(rounding){
        case ROUND_HALF_EVEN:
            return std::nearbyint(value);
        case ROUND_HALF_UP:
            return floor(value + .5);
        case ROUND_HALF_DOWN:
            return ceil(value - .5);
    }
    return 0;
}

template<typename Dtype>
__device__ inline
Dtype CLIP(const Dtype v, const Dtype min, const Dtype max){
    if(v > max) return max;
    if(v < min) return min;
    return v;
}

template<typename Dtype, typename Stype, typename ZPtype>
__device__ __inline__
int QuantizeScalar(
    const Dtype value, const Stype scale, const ZPtype zero_point,
    const int clip_min, const int clip_max,
    const Rounding rounding){
    /**
     * PPQ Quantization Function implementation.
     * This function convert an float value to int32
     *
     * Say we have a float value f, and int value i
     * This Transformation satisfies: f = clip(i / s + o)
     * Where s is scale factor, and o is offset
     */
    int v = _round2int(value / scale, rounding) + zero_point;
    return CLIP<int>(v, clip_min, clip_max);
}

template<typename Dtype, typename Stype, typename ZPtype>
__device__ __inline__
float DequantizeScalar(
    const Dtype value, const Stype scale, const ZPtype zero_point){
    /**
     * PPQ Quantization Function implementation.
     * This function convert an int32 value to float
     *
     * Say we have a float value f, and int value i
     * This Transformation satisfies: f = (i - o) * s
     * Where s is scale factor, and o is offset
     */
    return (value - zero_point) * scale;
}
