#include <torch/all.h>
#include <ATen/ATen.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include "tokenwise_quant.cuh"

std::vector<torch::Tensor> quant_pertoken_cuda(torch::Tensor &input)
{
    const int size = input.numel();
    const int col = input.size(-1);
    const int row = size / col;
    auto output = torch::empty(input.sizes(), torch::TensorOptions().dtype(torch::kChar).device(torch::kCUDA)).contiguous();
    TORCH_CHECK(output.size(-1) % 4 == 0, "a pack of 4-element is required");
    auto shapes = input.sizes().vec();
    shapes.pop_back();
    auto inv_scales = torch::empty(at::IntArrayRef(shapes), torch::TensorOptions().dtype(input.dtype()).device(torch::kCUDA)).contiguous();

    if (input.dtype() == torch::kFloat)
    {
        const float *inp = input.data_ptr<float>();
        float *inv_scale = inv_scales.data_ptr<float>();
        int8_t *out = output.data<int8_t>();
        using input_dtype = float;
        using output_dtype = int8_t;
        cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
        tokenwise_quant::DirectLoad<input_dtype, float> load(inp, col);
        tokenwise_quant::DirectStore<int8_t, output_dtype> store(out, col);
        tokenwise_quant::DirectStore<float, input_dtype> scale_store(inv_scale, 1);
        tokenwise_quant::DispatchQuant<decltype(load), decltype(store), decltype(scale_store), float>(
            cuda_stream, load, store, scale_store, row, col);
    }
    else if (input.dtype() == torch::kHalf)
    {
        const __half *inp = reinterpret_cast<__half *>(input.data_ptr<at::Half>());
        __half *inv_scale = reinterpret_cast<__half *>(inv_scales.data_ptr<at::Half>());
        int8_t *out = output.data<int8_t>();
        using input_dtype = __half;
        using output_dtype = int8_t;
        cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
        tokenwise_quant::DirectLoad<input_dtype, float> load(inp, col);
        tokenwise_quant::DirectStore<int8_t, output_dtype> store(out, col);
        tokenwise_quant::DirectStore<float, input_dtype> scale_store(inv_scale, 1);
        tokenwise_quant::DispatchQuant<decltype(load), decltype(store), decltype(scale_store), float>(
            cuda_stream, load, store, scale_store, row, col);
    }
    else
    {
        TORCH_CHECK(false, "No implementations supplied for input dtype.");
    }
    return std::vector<torch::Tensor>{output, inv_scales};
}
