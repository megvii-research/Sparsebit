#include <torch/all.h>
#include <torch/types.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

//======================== 4 bit ========================
void vecquant4matmul_cuda(
    torch::Tensor inp1, torch::Tensor inp2, torch::Tensor out,
    torch::Tensor scales, torch::Tensor zeros, int group_size = 0);

void vecquant4matmulbackward_cuda(
    torch::Tensor inp1, torch::Tensor inp2, torch::Tensor out,
    torch::Tensor scales, torch::Tensor zeros, int group_size = 0);

void vecquant4matmul(
    torch::Tensor inp1, torch::Tensor inp2, torch::Tensor out,
    torch::Tensor scales, torch::Tensor zeros)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(inp1));
    vecquant4matmul_cuda(inp1, inp2, out, scales, zeros, 0);
}
void vecgroupquant4matmul(
    torch::Tensor inp1, torch::Tensor inp2, torch::Tensor out,
    torch::Tensor scales, torch::Tensor zeros, int group_size)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(inp1));
    vecquant4matmul_cuda(inp1, inp2, out, scales, zeros, group_size);
}

void vecquant4matmulbackward(
    torch::Tensor inp1, torch::Tensor inp2, torch::Tensor out,
    torch::Tensor scales, torch::Tensor zeros)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(inp1));
    vecquant4matmulbackward_cuda(inp1, inp2, out, scales, zeros, 0);
}
void vecgroupquant4matmulbackward(
    torch::Tensor inp1, torch::Tensor inp2, torch::Tensor out,
    torch::Tensor scales, torch::Tensor zeros, int group_size)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(inp1));
    vecquant4matmulbackward_cuda(inp1, inp2, out, scales, zeros, group_size);
}

//======================== unpack ======================
at::Tensor unpack_cuda(at::Tensor &inputs, at::Tensor &zeros, bool transpose);
at::Tensor unpack(at::Tensor inputs, at::Tensor zeros, bool transpose = false)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(inputs));
    return unpack_cuda(inputs, zeros, transpose);
}
at::Tensor unpack_backward_cuda(at::Tensor &inputs, at::Tensor &scales, at::Tensor &zeros, at::Tensor &ic_scales, at::Tensor &ic_zeros, bool transpose);
at::Tensor unpack_backward(at::Tensor &inputs, at::Tensor &scales, at::Tensor &zeros, at::Tensor &ic_scales, at::Tensor &ic_zeros, bool transpose = false)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(inputs));
    return unpack_backward_cuda(inputs, scales, zeros, ic_scales, ic_zeros, transpose);
}
//======================== quant ======================
std::vector<torch::Tensor> quant_pertoken_cuda(torch::Tensor &input);
std::vector<torch::Tensor> quant_pertoken(at::Tensor inputs)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(inputs));
    return quant_pertoken_cuda(inputs);
}

//======================== cutlass ====================
void int8gemm_cuda(
    torch::Tensor inputs, torch::Tensor weights, torch::Tensor out, // torch::Tensor bias,
    float alpha, float beta);

void int8gemm(
    torch::Tensor inputs, torch::Tensor weights, torch::Tensor out, // torch::Tensor bias,
    float alpha, float beta)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(inputs));
    int8gemm_cuda(inputs, weights, out, alpha, beta);
}

//======================== pybind =======================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("vecquant4matmul", &vecquant4matmul, "Vector 4-bit Quantized Matrix Multiplication (CUDA)");
    m.def("unpack", &unpack, "Unpack 8-bit into two 4-bits");
    m.def("unpack_backward", &unpack_backward, "Unpack, and requant to 8-bit");
    m.def("int8gemm", &int8gemm, "the kernel implement vias cutlass");
    m.def("quant_pertoken", &quant_pertoken, "per-token quantization");
}
