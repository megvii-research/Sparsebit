#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void VecQuant4MatMulKernel(
    const scalar_t *__restrict__ inp1,
    const int *__restrict__ inp2,
    scalar_t *__restrict__ out,
    const scalar_t *__restrict__ scales,
    const scalar_t *__restrict__ zeros,
    int height,
    int width,
    int inchannels);

const int BLOCKWIDTH_4BIT = 1024;
const int BLOCKHEIGHT_4BIT = 128;

#define BLOCKWIDTH BLOCKWIDTH_4BIT
#define BLOCKHEIGHT BLOCKHEIGHT_4BIT

void vecquant4matmul_cuda(
    torch::Tensor inp1,
    torch::Tensor inp2,
    torch::Tensor out,
    torch::Tensor scales,
    torch::Tensor zeros)
{
    int height = inp2.size(0);
    int width = inp2.size(1);
    int inchannels = inp1.size(-1);

    dim3 blocks(
        (height + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
        (width + BLOCKWIDTH - 1) / BLOCKWIDTH); // ceil(M/128), ceil(N/1024)
    dim3 threads(BLOCKWIDTH);

    AT_DISPATCH_FLOATING_TYPES(
        inp1.type(), "vecquant4matmul_cuda",
        ([&]
         { VecQuant4MatMulKernel<<<blocks, threads>>>(
               inp1.data<scalar_t>(),
               inp2.data<int>(),
               out.data<scalar_t>(),
               scales.data<scalar_t>(),
               zeros.data<scalar_t>(),
               height,
               width,
               inchannels); }));
}

__device__ inline unsigned int as_unsigned(int i)
{
    return *reinterpret_cast<unsigned int *>(&i);
}

template <typename scalar_t>
__global__ void VecQuant4MatMulKernel(
    const scalar_t *__restrict__ inp1,
    const int *__restrict__ inp2,
    scalar_t *__restrict__ out,
    const scalar_t *__restrict__ scales,
    const scalar_t *__restrict__ zeros,
    int height,
    int width,
    int inchannels)
{

    int row = BLOCKHEIGHT * blockIdx.x;
    int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
    int inp1_pos = BLOCKWIDTH * blockIdx.x + threadIdx.x;

    __shared__ scalar_t tmps[BLOCKWIDTH];
    if (inp1_pos < inchannels)
        tmps[threadIdx.x] = inp1[inp1_pos];
    else
        tmps[threadIdx.x] = 0;
    __syncthreads();

    if (col < width)
    {
        scalar_t scale = scales[col];
        scalar_t zero = zeros[col];

        scalar_t res = 0;
        int i = width * row + col, max_i = height * width;
        int k = 0;

        unsigned int tmp1;

        while (k < BLOCKWIDTH && i < max_i)
        {
            tmp1 = as_unsigned(inp2[i]);
            res += (scale * scalar_t((tmp1 >> 0) & 0xF) - zero) * tmps[k + 0];
            res += (scale * scalar_t((tmp1 >> 4) & 0xF) - zero) * tmps[k + 1];
            res += (scale * scalar_t((tmp1 >> 8) & 0xF) - zero) * tmps[k + 2];
            res += (scale * scalar_t((tmp1 >> 12) & 0xF) - zero) * tmps[k + 3];
            res += (scale * scalar_t((tmp1 >> 16) & 0xF) - zero) * tmps[k + 4];
            res += (scale * scalar_t((tmp1 >> 20) & 0xF) - zero) * tmps[k + 5];
            res += (scale * scalar_t((tmp1 >> 24) & 0xF) - zero) * tmps[k + 6];
            res += (scale * scalar_t((tmp1 >> 28) & 0xF) - zero) * tmps[k + 7];
            i += width;
            k += 8;
        }
        atomicAdd(&out[col], res);
    }
}

#undef BLOCKWIDTH
#undef BLOCKHEIGHT
