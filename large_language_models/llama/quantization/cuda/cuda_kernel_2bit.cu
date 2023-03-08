#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void VecQuant2MatMulKernel(
    const scalar_t *__restrict__ inp1,
    const int *__restrict__ inp2,
    scalar_t *__restrict__ out,
    const scalar_t *__restrict__ scales,
    const scalar_t *__restrict__ zeros,
    int height,
    int width,
    int inchannels);

const int BLOCKWIDTH_2BIT = 1024;
const int BLOCKHEIGHT_2BIT = 64;

#define BLOCKWIDTH BLOCKWIDTH_2BIT
#define BLOCKHEIGHT BLOCKHEIGHT_2BIT

void vecquant2matmul_cuda(
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
        (width + BLOCKWIDTH - 1) / BLOCKWIDTH);
    dim3 threads(BLOCKWIDTH);
    AT_DISPATCH_FLOATING_TYPES(
        inp1.type(), "vecquant2matmul_cuda",
        ([&]
         { VecQuant2MatMulKernel<<<blocks, threads>>>(
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
__global__ void VecQuant2MatMulKernel(
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

    __shared__ scalar_t blockvec[BLOCKWIDTH];
    if (inp1_pos < inchannels)
        blockvec[threadIdx.x] = inp1[inp1_pos];
    else
        blockvec[threadIdx.x] = 0;
    __syncthreads();

    if (col < width)
    {
        scalar_t scale = scales[col];
        scalar_t zero = zeros[col];

        scalar_t res = 0;
        int i = width * row + col, max_i = height * width;
        int k = 0;
        //int bit = 2;
        //int block_shift = 32 / bit;
        unsigned int tmp1;
        while (k < BLOCKWIDTH && i < max_i)
        {
            tmp1 = as_unsigned(inp2[i]);
            for (int shift = 0; shift < 16; shift++){
                res += (scale * scalar_t((tmp1 >> (2*shift)) & 0x3) - zero) * blockvec[k + shift];
            }
            i += width; // 为什么要跳到下一行? 因为要切换w
            k += 16;
        }
        

        atomicAdd(&out[col], res);
    }
}

#undef BLOCKWIDTH
#undef BLOCKHEIGHT
