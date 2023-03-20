#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

//======================== 4 bit ========================
void vecquant4matmul_cuda(
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

//======================== 3 bit ========================
void vecquant3matmul_cuda(
    torch::Tensor inp1, torch::Tensor inp2, torch::Tensor out,
    torch::Tensor scales, torch::Tensor zeros, int group_size = 0);

void vecquant3matmul(
    torch::Tensor inp1, torch::Tensor inp2, torch::Tensor out,
    torch::Tensor scales, torch::Tensor zeros)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(inp1));
    vecquant3matmul_cuda(inp1, inp2, out, scales, zeros);
}
void vecgroupquant3matmul(
    torch::Tensor inp1, torch::Tensor inp2, torch::Tensor out,
    torch::Tensor scales, torch::Tensor zeros, int group_size)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(inp1));
    vecquant3matmul_cuda(inp1, inp2, out, scales, zeros, group_size);
}

//======================== 2 bit ========================
void vecquant2matmul_cuda(
    torch::Tensor inp1, torch::Tensor inp2, torch::Tensor out,
    torch::Tensor scales, torch::Tensor zeros, int group_size = 0);

void vecquant2matmul(
    torch::Tensor inp1, torch::Tensor inp2, torch::Tensor out,
    torch::Tensor scales, torch::Tensor zeros)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(inp1));
    vecquant2matmul_cuda(inp1, inp2, out, scales, zeros);
}
void vecgroupquant2matmul(
    torch::Tensor inp1, torch::Tensor inp2, torch::Tensor out,
    torch::Tensor scales, torch::Tensor zeros, int group_size)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(inp1));
    vecquant2matmul_cuda(inp1, inp2, out, scales, zeros, group_size);
}

//======================== pybind =======================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("vecquant2matmul", &vecquant2matmul, "Vector 2-bit Quantized Matrix Multiplication (CUDA)");
    m.def("vecquant3matmul", &vecquant3matmul, "Vector 3-bit Quantized Matrix Multiplication (CUDA)");
    m.def("vecquant4matmul", &vecquant4matmul, "Vector 4-bit Quantized Matrix Multiplication (CUDA)");
    m.def("vecgroupquant2matmul", &vecgroupquant2matmul, "Vector 2-bit Group Quantized Matrix Multiplication (CUDA)");
    m.def("vecgroupquant3matmul", &vecgroupquant3matmul, "Vector 3-bit Group Quantized Matrix Multiplication (CUDA)");
    m.def("vecgroupquant4matmul", &vecgroupquant4matmul, "Vector 4-bit Group Quantized Matrix Multiplication (CUDA)");
}
