#include <torch/all.h>
#include <torch/python.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int BLOCKSIZE = 256;

__global__ void unpack_kernel(
    const int8_t *__restrict__ inputs,
    const int8_t *__restrict__ zeros,
    int8_t *__restrict__ outputs,
    const int ic,
    const int oc,
    bool transpose);

torch::Tensor unpack_cuda(torch::Tensor &inputs, torch::Tensor &zeros, bool transpose = false)
{
    // the layout inputs is (ic, oc)
    const int inputs_nrows = inputs.size(0);
    const int inputs_ncols = inputs.size(1);
    int output_nrows = 0;
    int output_ncols = 0;

    if (!transpose)
    {
        output_nrows = 2 * inputs_nrows;
        output_ncols = inputs_ncols;
    }
    else
    {
        output_nrows = inputs_ncols;
        output_ncols = 2 * inputs_nrows;
    }

    auto outputs = at::empty({output_nrows, output_ncols}, inputs.options());

    const int n = inputs.numel();
    dim3 blocks((n + BLOCKSIZE - 1) / BLOCKSIZE); // 按256切线程?
    dim3 threads(BLOCKSIZE);

    unpack_kernel<<<blocks, threads>>>(
        inputs.data<int8_t>(),
        zeros.data<int8_t>(),
        outputs.data<int8_t>(),
        inputs_nrows,
        inputs_ncols,
        transpose);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error: %s", cudaGetErrorString(err));
    }

    return outputs;
}

__global__ void unpack_kernel(
    const int8_t *__restrict__ inputs,
    const int8_t *__restrict__ zeros,
    int8_t *__restrict__ outputs,
    const int inputs_nrows,
    const int inputs_ncols,
    bool transpose)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int8_t val = inputs[idx];
    if (idx >= inputs_nrows * inputs_ncols)
        return;
    int i = idx / inputs_ncols; // 第几行
    int j = idx % inputs_ncols; // 第几列
    int out_idx, out_idx2;
    if (!transpose)
    {
        out_idx = i * 2 * inputs_ncols + j;
        out_idx2 = (i * 2 + 1) * inputs_ncols + j;
    }
    else
    {
        out_idx = j * 2 * inputs_nrows + 2 * i;
        out_idx2 = j * 2 * inputs_nrows + 2 * i + 1;
    }
    outputs[out_idx] = (val & 0xF) - zeros[j];
    outputs[out_idx2] = ((val >> 4) & 0xF) - zeros[j];
}

#define SZ 32
#define DIVUP(x, y) (((x) + (y)-1) / (y))

//=======================================================
template <typename scalar_t>
__global__ void unpack_backward_kernel(
    const int8_t *__restrict__ inputs,
    const scalar_t *__restrict__ oc_scales,
    const scalar_t *__restrict__ oc_zeros,
    const scalar_t *__restrict__ ic_scales,
    const scalar_t *__restrict__ ic_zeros,
    int8_t *__restrict__ outputs,
    const int ic,
    const int oc,
    bool transpose);

torch::Tensor unpack_backward_cuda(torch::Tensor &inputs, torch::Tensor &scales, torch::Tensor &zeros, torch::Tensor &ic_scales, torch::Tensor &ic_zeros, bool transpose = false)
{
    // the layout inputs is (ic, oc)
    const int inputs_nrows = inputs.size(0);
    const int inputs_ncols = inputs.size(1);
    int output_nrows = 0;
    int output_ncols = 0;

    TORCH_CHECK(scales.numel() == inputs_ncols);
    TORCH_CHECK(zeros.numel() == inputs_ncols);
    TORCH_CHECK(ic_scales.numel() == inputs_nrows * 2);
    TORCH_CHECK(ic_zeros.numel() == inputs_nrows * 2);
    if (!transpose)
    {
        output_nrows = 2 * inputs_nrows;
        output_ncols = inputs_ncols;
    }
    else
    {
        output_nrows = inputs_ncols;
        output_ncols = 2 * inputs_nrows;
    }

    auto outputs = at::empty({output_nrows, output_ncols}, inputs.options());

    dim3 blocks(DIVUP(inputs_nrows, SZ), DIVUP(inputs_ncols, SZ));
    dim3 threads(SZ);

    if (scales.dtype() == torch::kFloat)
    {
        using scalar_t = float;
        unpack_backward_kernel<scalar_t><<<blocks, threads>>>(
            inputs.data<int8_t>(),
            scales.data<scalar_t>(),
            zeros.data<scalar_t>(),
            ic_scales.data<scalar_t>(),
            ic_zeros.data<scalar_t>(),
            outputs.data<int8_t>(),
            inputs_nrows,
            inputs_ncols,
            transpose);
    }
    else if (scales.dtype() == torch::kHalf)
    {
        using scalar_t = __half;
#define ptr(x) reinterpret_cast<scalar_t *>(x.data<at::Half>())
        unpack_backward_kernel<scalar_t><<<blocks, threads>>>(
            inputs.data<int8_t>(),
            ptr(scales),
            ptr(zeros),
            ptr(ic_scales),
            ptr(ic_zeros),
            outputs.data<int8_t>(),
            inputs_nrows,
            inputs_ncols,
            transpose);
#undef ptr
    }

    return outputs;
}

__device__ __forceinline__ int8_t to_int8(const float x)
{
    return min(max(__float2int_rn(x), -127), 127);
}
__device__ __forceinline__ int8_t to_int8(const __half x)
{
    return min(max(__half2int_rn(x), -127), 127);
}

template <typename scalar_t>
__global__ void unpack_backward_kernel(
    const int8_t *__restrict__ inputs,
    const scalar_t *__restrict__ oc_scales,
    const scalar_t *__restrict__ oc_zeros,
    const scalar_t *__restrict__ ic_scales,
    const scalar_t *__restrict__ ic_zeros,
    int8_t *__restrict__ outputs,
    const int ic,
    const int oc,
    bool transpose)
{
    int bx = blockIdx.x * SZ, by = blockIdx.y * SZ;
    int idx = threadIdx.x;
    scalar_t oc_scale[SZ], oc_zero[SZ], ic_scale[SZ << 1], ic_zero[SZ << 1];
    for (int i = bx * 2, j = 0; i < ic * 2 && j < (SZ * 2); i += 1, j += 1)
    {
        ic_scale[j] = ic_scales[i];
        ic_zero[j] = ic_zeros[i];
    }
    for (int i = by, j = 0; i < oc && j < SZ; i += 1, j += 1)
    {
        oc_scale[j] = oc_scales[i];
        oc_zero[j] = oc_zeros[i];
    }
    int pos0 = bx * oc + by + idx, pos1, stride1;
    int IC = 2 * ic, OC = oc;
    if (!transpose)
    {
        pos1 = bx * 2 * OC + by + idx;
        stride1 = OC;
    }
    else
    {
        pos1 = (by + idx) * IC + bx * 2;
        stride1 = 1;
    }
    int8_t unpacked_w;
    scalar_t mid_val;
    if (by + idx < oc)
    {
        for (int i = 0; i < SZ && bx + i < ic; i++)
        {
            unpacked_w = inputs[pos0];
            mid_val = scalar_t(unpacked_w & 0xF) * oc_scale[idx] - oc_zero[idx];
            outputs[pos1] = to_int8((mid_val + ic_zero[i << 1]) / ic_scale[i << 1]);
            pos1 += stride1;
            mid_val = scalar_t((unpacked_w & 0xF0) >> 4) * oc_scale[idx] - oc_zero[idx];
            outputs[pos1] = to_int8((mid_val + ic_zero[i << 1 | 1]) / ic_scale[i << 1 | 1]);
            pos1 += stride1;
            pos0 += oc;
        }
    }
}
