#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>

using at::cuda::getCurrentCUDAStream;

template <typename scalar_t>
__global__ void VecQuant2MatMulKernel(
    const scalar_t *__restrict__ inp1,
    const int *__restrict__ inp2,
    scalar_t *__restrict__ out,
    const scalar_t *__restrict__ scales,
    const scalar_t *__restrict__ zeros,
    int height,
    int width,
    int inchannels,
    int group_size);

const int BLOCKWIDTH_2BIT = 1024;
const int BLOCKHEIGHT_2BIT = 64;

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVUP(a, b) (((a) + (b)-1) / (b))
#define BLOCKWIDTH BLOCKWIDTH_2BIT
#define BLOCKHEIGHT BLOCKHEIGHT_2BIT
#define BIT 2

#define DIM_LEN(x) ((x).sizes().vec().size())

void vecquant2matmul_cuda(
    torch::Tensor inp1,
    torch::Tensor inp2,
    torch::Tensor out,
    torch::Tensor scales,
    torch::Tensor zeros,
    int group_size = 0)
{
    TORCH_CHECK(DIM_LEN(inp1) >= 2, "input1 must be with dimension > 2");
    int inchannels = inp1.size(-1);
    int batch = inp1.numel() / inchannels;

    TORCH_CHECK(DIM_LEN(inp2) == 2, "input2 must be with dimension == 2");
    int height = inp2.size(0);
    int width = inp2.size(1);

    TORCH_CHECK(out.size(-1) == width, "output channel must be the same with input2 out_channel");

    dim3 blocks(
        batch,
        (height + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
        (width + BLOCKWIDTH - 1) / BLOCKWIDTH);
    dim3 threads(BLOCKWIDTH);

    if (group_size != 0)
    {
        TORCH_CHECK(group_size / 64 * 64 == group_size, "only group_size divisible by 64 is supported in 2-bit quantization");
    }
    else
        group_size = inchannels;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(
        inp1.type(), "vecquant2matmul_cuda",
        ([&]
         { VecQuant2MatMulKernel<<<blocks, threads, (BLOCKWIDTH + BLOCKWIDTH / 64 * 2) * sizeof(scalar_t), stream>>>(
               inp1.data<scalar_t>(),
               inp2.data<int>(),
               out.data<scalar_t>(),
               scales.data<scalar_t>(),
               zeros.data<scalar_t>(),
               height,
               width,
               inchannels,
               group_size); }));
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
    int inchannels,
    int group_size)
{
    int cur_batch = blockIdx.x;
    int row = BLOCKHEIGHT * blockIdx.y;
    int col = BLOCKWIDTH * blockIdx.z + threadIdx.x;
    int inp1_pos = BLOCKWIDTH * blockIdx.y + threadIdx.x;
    int group_pos = row * 32 / BIT / group_size;
    int group_max = DIVUP(inchannels, group_size);

    __shared__ scalar_t blockvec[BLOCKWIDTH];
    if (inp1_pos < inchannels)
        blockvec[threadIdx.x] = inp1[cur_batch * inchannels + inp1_pos];
    else
        blockvec[threadIdx.x] = 0;

    scalar_t scale_cache[BLOCKWIDTH / 64] = {0};
    scalar_t zero_cache[BLOCKWIDTH / 64] = {0};
    if (col < width)
    {
        scale_cache[0] = scales[group_max * col + group_pos];
        zero_cache[0] = zeros[group_max * col + group_pos];
        for (int i = 1; i < BLOCKWIDTH / 64; ++i)
            if (group_pos + i < group_max)
            {
                scale_cache[i] = scales[group_max * col + group_pos + i];
                zero_cache[i] = zeros[group_max * col + group_pos + i];
            }
    }
    __syncthreads();

    if (col < width)
    {
        scalar_t scale;
        scalar_t zero;

        scalar_t res = 0;
        int i = width * row + col, max_i = height * width;
        int k = 0;
        int p = row * 32 / BIT;

        unsigned int tmp1;
        while (k < BLOCKWIDTH && i < max_i)
        {
            scale = scale_cache[p / group_size - group_pos];
            zero = zero_cache[p / group_size - group_pos];
            tmp1 = as_unsigned(inp2[i]);
            for (int shift = 0; shift < 16; shift++)
                res += (scale * scalar_t((tmp1 >> (2 * shift)) & 0x3) - zero) * blockvec[k + shift];
            i += width; // 为什么要跳到下一行? 因为要切换w
            k += 16;
            p += 16;
        }

        atomicAdd(&out[cur_batch * width + col], res);
    }
}

#undef BLOCKWIDTH
#undef BLOCKHEIGHT
