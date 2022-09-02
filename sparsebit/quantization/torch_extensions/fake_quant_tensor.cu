/*
Code are based on 
https://github.com/openppl-public/ppq/blob/deec01b14a61cdd72627fc0de03882c699723e76/ppq/csrc/cuda/linear.cu
Copyright(C) OpenPPL, Apache License 2.0
*/


# include "common.cuh"
# include "fake_quant_tensor.h"


template<typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask,
                                            int width = 32, unsigned int mask = FINAL_MASK) {
#if __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 9000
    return __shfl_xor_sync(mask, value, laneMask, width);
#else
    return __shfl_xor(value, laneMask, width);
#endif
}


template<typename T>
__device__ __forceinline__ T WarpReduceSum(T val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val += WARP_SHFL_XOR(val, mask, 32, FINAL_MASK);
    return val;
}


template<typename T>
__device__ __forceinline__ T BlockReduceSum(T val) {
    static __shared__ T shared[32];
    if(threadIdx.x < 32) shared[threadIdx.x] = 0;
    __syncthreads();
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = WarpReduceSum(val);
    if(lane == 0) shared[wid] = val;
    __syncthreads();

    // 这里有问题
    val = (lane < (blockDim.x >> 5)) ? shared[lane] : (T)0.0f;
    val = WarpReduceSum(val);
    return val;
}


__global__ void QuantizePerTensorForwardCUDA(
    const int64_t num_elements,
    const float* data,
    const float* scale,
    const float* zero_point,
    const int qmin,
    const int qmax,
    const Rounding rounding,
    float* out){
    float s = scale[0]; int zp = std::round(zero_point[0]); 
    for(int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elements; i += blockDim.x*gridDim.x){
        int vq = _round2int(data[i] / s, rounding) + zp;
        vq = CLIP<int>(vq, qmin, qmax);
        float vfq = (vq - zp) * s;
        out[i] = vfq;
    }
}


__host__ Tensor QuantizePerTensorForward(
    const Tensor &data, 
    const Tensor &scale,
    const Tensor &zero_point,
    const int qmin,
    const int qmax,
    const Rounding rounding){
    CheckTensor(data, at::kFloat, "data(Expect to be FP32)");
    CheckTensor(scale, at::kFloat, "scale(Expect to be FP32)");
    CheckTensor(zero_point, at::kFloat, "zero_point(Expect to be FP32)");

    Tensor data_fq = at::empty_like(data);
    int64_t num_elements = data.numel();

    QuantizePerTensorForwardCUDA<<<NUM_OF_BLOCK(num_elements), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_elements,
        data.data_ptr<float>(),
        scale.data_ptr<float>(),
        zero_point.data_ptr<float>(),
        qmin,
        qmax,
        rounding,
        data_fq.data_ptr<float>());

    return data_fq;
}


__global__ void QuantizePerTensorBackwardCUDA(
    const int64_t num_elements,
    const float* data, 
    const float* scale,
    const float* zero_point,
    const float* grad_y,
    const int qmin,
    const int qmax,
    const Rounding rounding,
    float* gx,
    float* gs,
    float* gzp,
    const bool enable_gs,
    const bool enable_gzp){
    float s = scale[0]; int zp = std::round(zero_point[0]);
    for(int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elements; i += blockDim.x*gridDim.x){
        float x = data[i];
        float gy = grad_y[i];
        int vq = _round2int(x / s, rounding) + zp;
        float partial_gx = gy;
        if (vq > qmax || vq < qmin) partial_gx = 0;
        gx[i] = partial_gx;
        if(enable_gs){
            float partial_gs = (_round2int(x/s, rounding) - x/s) * gy;
            if(vq > qmax) partial_gs = (qmax - zp) * gy;
            if(vq < qmin) partial_gs = (qmin - zp) * gy;
            float reduced_gs = BlockReduceSum<float>(partial_gs); __syncthreads();
            if (threadIdx.x == 0) atomicAdd(gs, reduced_gs);
        }
        if(enable_gzp){
            float partial_gzp = (vq <= qmax && vq >= qmin) ? 0.0f : (-s * gy);
            float reduced_gzp = BlockReduceSum<float>(partial_gzp); __syncthreads();
            if (threadIdx.x == 0) atomicAdd(gzp, reduced_gzp);
        }
    }
}


__host__ std::vector<Tensor> QuantizePerTensorBackward(
    const Tensor &data, const Tensor &scale, const Tensor &zero_point, const Tensor &grad_y,
    const int qmin, const int qmax, const Rounding rounding){
    /**
     * Gradient Bakcwrad for quantization
     * Solve grad_s, grad_o, grad_v at once.
     */
    CheckTensor(data, at::kFloat, "data(Expect to be FP32)");
    CheckTensor(scale, at::kFloat, "scale(Expect to be FP32)");
    CheckTensor(zero_point, at::kFloat, "zero_point(Expect to be FP32)");
    CheckTensor(grad_y, at::kFloat, "Gard(Expect to be FP32)");

    Tensor grad_x = at::zeros_like(data);
    Tensor grad_s = at::zeros_like(scale);
    Tensor grad_zp = at::zeros_like(zero_point);

    int64_t num_elements = data.numel();
    QuantizePerTensorBackwardCUDA<<<NUM_OF_BLOCK(num_elements), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_elements,
        data.data_ptr<float>(),
        scale.data_ptr<float>(),
        zero_point.data_ptr<float>(),
        grad_y.data_ptr<float>(),
        qmin,
        qmax,
        rounding,
        grad_x.data_ptr<float>(),
        grad_s.data_ptr<float>(),
        grad_zp.data_ptr<float>(),
        scale.requires_grad(),
        zero_point.requires_grad());
    return {grad_x, grad_s, grad_zp};
}


__global__ void QuantizePerChannelForwardCUDA(
    const int64_t num_elements,
    const int64_t num_elements_perchannel,
    const int num_channels,
    const float* data,
    const float* scale,
    const float* zero_point,
    const int qmin,
    const int qmax,
    const Rounding rounding,
    float* out){
    for(int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elements; i += blockDim.x*gridDim.x){
        int c = (i / num_elements_perchannel) % num_channels;
        float s = scale[c]; int zp = std::round(zero_point[c]);
        auto x_q = QuantizeScalar<float, float, int>(data[i], s, zp, qmin, qmax, rounding);
        float x_fq = DequantizeScalar<int, float, int>(x_q, s, zp);
        out[i] = x_fq;
    }
}


__host__ Tensor QuantizePerChannelForward(
    const Tensor &data, 
    const Tensor &scale,
    const Tensor &zero_point,
    const int qmin, 
    const int qmax, 
    const int ch_axis, 
    const Rounding rounding){
    CheckTensor(data, at::kFloat, "data(Expect to be FP32)");
    CheckTensor(scale, at::kFloat, "Scale(Expect to be FP32)");
    CheckTensor(zero_point, at::kFloat, "ZeroPoint(Expect to be FP32)");

    int num_elements_perchannel = 1;
    const int num_channels = data.sizes()[ch_axis];
    // incompatible transformer, which layout is NLC
    for(int axis = data.ndimension() - 1; axis != ch_axis; axis--){
        num_elements_perchannel *= data.sizes()[axis];
    }

    int64_t num_elements = data.numel();
    Tensor data_fq = at::empty_like(data);
    QuantizePerChannelForwardCUDA<<<NUM_OF_BLOCK(num_elements), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_elements,
        num_elements_perchannel,
        num_channels,
        data.data_ptr<float>(),
        scale.data_ptr<float>(),
        zero_point.data_ptr<float>(),
        qmin,
        qmax,
        rounding,
        data_fq.data_ptr<float>());
    return data_fq;
}


__global__ void QuantizePerChannelBackwardCUDA(
    const int64_t num_elements,
    const int64_t num_elements_perchannel,
    const int num_channels,
    const float* data, 
    const float* scale,
    const float* zero_point,
    const float* grad_y,
    const int qmin,
    const int qmax,
    const Rounding rounding,
    float* gx,
    float* gs,
    float* gzp,
    const bool enable_gs,
    const bool enable_gzp){
    const int64_t num_elements_persample = num_elements_perchannel * num_channels;
    int c = blockIdx.x; float s = scale[c]; int zp = std::round(zero_point[c]);
    for(int64_t i = (c * num_elements_perchannel) + blockIdx.y * CUDA_NUM_THREADS + threadIdx.x; i < num_elements; i += num_elements_persample){
        float partial_gs = 0; float partial_gzp = 0;
        if (blockIdx.y * CUDA_NUM_THREADS + threadIdx.x < num_elements_perchannel){
            float x = data[i];
            float gy = grad_y[i];
            int vq = _round2int(x/s, rounding) + zp;
            // the graident of x
            float partial_gx = gy;
            if(vq > qmax || vq < qmin) partial_gx = 0;
            gx[i] = partial_gx;
            // the graident of scale
            if(enable_gs){
                partial_gs = (_round2int(x/s, rounding) - x/s) * grad_y[i];
                if(vq < qmin) partial_gs = (qmin - zp) * grad_y[i];
                if(vq > qmax) partial_gs = (qmax - zp) * grad_y[i];
                float reduced_gs = BlockReduceSum<float>(partial_gs); __syncthreads();
                if (threadIdx.x == 0) atomicAdd(&gs[c], reduced_gs);
            }
            if(enable_gzp){
                partial_gzp = (vq >= qmin && vq < qmax) ? 0.0f : (-s * grad_y[i]);
                float reduced_gzp = BlockReduceSum<float>(partial_gzp); __syncthreads();
                if (threadIdx.x == 0) atomicAdd(&gzp[c], reduced_gzp);
            }
        }
    }
}


__host__ std::vector<Tensor> QuantizePerChannelBackward(
    const Tensor &data, const Tensor &scale,
    const Tensor &zero_point, const Tensor &grad_y,
    const int qmin, const int qmax, const int ch_axis, const Rounding rounding){
    CheckTensor(data, at::kFloat, "Data(Expect to be FP32)");
    CheckTensor(scale, at::kFloat, "Scale(Expect to be FP32)");
    CheckTensor(zero_point, at::kFloat, "ZeroPoint(Expect to be FP32)");
    CheckTensor(grad_y, at::kFloat, "Gard(Expect to be FP32)");

    Tensor grad_x = at::zeros_like(data);
    Tensor grad_s = at::zeros_like(scale);
    Tensor grad_zp = at::zeros_like(zero_point);

    int num_elements_perchannel = 1;
    const int num_channels = data.sizes()[ch_axis];
    for(int axis = data.ndimension() - 1; axis != ch_axis; axis--){
        num_elements_perchannel *= data.sizes()[axis];
    }

    int64_t num_elements = data.numel();
    dim3 grid;
    grid.x  = static_cast<unsigned int>(num_channels);
    grid.y  = static_cast<unsigned int>(NUM_OF_BLOCK(num_elements_perchannel));
    grid.z  = 1;
    QuantizePerChannelBackwardCUDA<<<grid, CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_elements,
        num_elements_perchannel,
        num_channels,
        data.data_ptr<float>(),
        scale.data_ptr<float>(),
        zero_point.data_ptr<float>(),
        grad_y.data_ptr<float>(),
        qmin,
        qmax,
        rounding,
        grad_x.data_ptr<float>(),
        grad_s.data_ptr<float>(),
        grad_zp.data_ptr<float>(),
        scale.requires_grad(),
        zero_point.requires_grad());
    return {grad_x, grad_s, grad_zp};
}

