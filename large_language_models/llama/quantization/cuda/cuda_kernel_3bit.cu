#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>

using at::cuda::getCurrentCUDAStream;

template <typename scalar_t>
__global__ void VecQuant3MatMulKernel(
    const scalar_t *__restrict__ inp1,
    const int *__restrict__ inp2,
    scalar_t *__restrict__ out,
    const scalar_t *__restrict__ scales,
    const scalar_t *__restrict__ zeros,
    int height,
    int width,
    int inchannels,
    int group_size);

const int BLOCKWIDTH_3BIT = 1024;
const int BLOCKHEIGHT_3BIT = 96;

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVUP(a, b) (((a) + (b)-1) / (b))
#define BLOCKWIDTH BLOCKWIDTH_3BIT
#define BLOCKHEIGHT BLOCKHEIGHT_3BIT
#define BIT 3

#define DIM_LEN(x) ((x).sizes().vec().size())

void vecquant3matmul_cuda(
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
        TORCH_CHECK(group_size / 128 * 128 == group_size, "only group_size divisible by 128 is supported in 3-bit quantization");
    }
    else
        group_size = inchannels;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(
        inp1.type(), "vecquant3matmul_cuda",
        ([&]
         { VecQuant3MatMulKernel<<<blocks, threads, (BLOCKWIDTH + BLOCKWIDTH / 128 * 2) * sizeof(scalar_t), stream>>>(
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
__global__ void VecQuant3MatMulKernel(
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

    scalar_t scale_cache[BLOCKWIDTH / 128] = {0};
    scalar_t zero_cache[BLOCKWIDTH / 128] = {0};
    if (col < width)
    {
        scale_cache[0] = scales[group_max * col + group_pos];
        zero_cache[0] = zeros[group_max * col + group_pos];
        for (int i = 1; i < BLOCKWIDTH / 128; ++i)
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
        unsigned int tmp2;
        unsigned int tmp;

        while (k < BLOCKWIDTH && i < max_i)
        {
            scale = scale_cache[p / group_size - group_pos];
            zero = zero_cache[p / group_size - group_pos];
            tmp1 = as_unsigned(inp2[i]);
            res += (scale * scalar_t((tmp1 >> 0) & 0x7) - zero) * blockvec[k + 0];
            res += (scale * scalar_t((tmp1 >> 3) & 0x7) - zero) * blockvec[k + 1];
            res += (scale * scalar_t((tmp1 >> 6) & 0x7) - zero) * blockvec[k + 2];
            res += (scale * scalar_t((tmp1 >> 9) & 0x7) - zero) * blockvec[k + 3];
            res += (scale * scalar_t((tmp1 >> 12) & 0x7) - zero) * blockvec[k + 4];
            res += (scale * scalar_t((tmp1 >> 15) & 0x7) - zero) * blockvec[k + 5];
            res += (scale * scalar_t((tmp1 >> 18) & 0x7) - zero) * blockvec[k + 6];
            res += (scale * scalar_t((tmp1 >> 21) & 0x7) - zero) * blockvec[k + 7];
            res += (scale * scalar_t((tmp1 >> 24) & 0x7) - zero) * blockvec[k + 8];
            res += (scale * scalar_t((tmp1 >> 27) & 0x7) - zero) * blockvec[k + 9];
            i += width;
            if (i >= max_i)
                tmp2 = 0;
            else
                tmp2 = as_unsigned(inp2[i]);
            tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
            tmp2 >>= 1;
            res += (scale * scalar_t(tmp) - zero) * blockvec[k + 10];
            k += 11;
            res += (scale * scalar_t((tmp2 >> 0) & 0x7) - zero) * blockvec[k + 0];
            res += (scale * scalar_t((tmp2 >> 3) & 0x7) - zero) * blockvec[k + 1];
            res += (scale * scalar_t((tmp2 >> 6) & 0x7) - zero) * blockvec[k + 2];
            res += (scale * scalar_t((tmp2 >> 9) & 0x7) - zero) * blockvec[k + 3];
            res += (scale * scalar_t((tmp2 >> 12) & 0x7) - zero) * blockvec[k + 4];
            res += (scale * scalar_t((tmp2 >> 15) & 0x7) - zero) * blockvec[k + 5];
            res += (scale * scalar_t((tmp2 >> 18) & 0x7) - zero) * blockvec[k + 6];
            res += (scale * scalar_t((tmp2 >> 21) & 0x7) - zero) * blockvec[k + 7];
            res += (scale * scalar_t((tmp2 >> 24) & 0x7) - zero) * blockvec[k + 8];
            res += (scale * scalar_t((tmp2 >> 27) & 0x7) - zero) * blockvec[k + 9];
            i += width;
            if (i >= max_i)
                tmp1 = 0;
            else
                tmp1 = as_unsigned(inp2[i]);
            tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
            tmp1 >>= 2;
            res += (scale * scalar_t(tmp) - zero) * blockvec[k + 10];
            k += 11;
            res += (scale * scalar_t((tmp1 >> 0) & 0x7) - zero) * blockvec[k + 0];
            res += (scale * scalar_t((tmp1 >> 3) & 0x7) - zero) * blockvec[k + 1];
            res += (scale * scalar_t((tmp1 >> 6) & 0x7) - zero) * blockvec[k + 2];
            res += (scale * scalar_t((tmp1 >> 9) & 0x7) - zero) * blockvec[k + 3];
            res += (scale * scalar_t((tmp1 >> 12) & 0x7) - zero) * blockvec[k + 4];
            res += (scale * scalar_t((tmp1 >> 15) & 0x7) - zero) * blockvec[k + 5];
            res += (scale * scalar_t((tmp1 >> 18) & 0x7) - zero) * blockvec[k + 6];
            res += (scale * scalar_t((tmp1 >> 21) & 0x7) - zero) * blockvec[k + 7];
            res += (scale * scalar_t((tmp1 >> 24) & 0x7) - zero) * blockvec[k + 8];
            res += (scale * scalar_t((tmp1 >> 27) & 0x7) - zero) * blockvec[k + 9];
            i += width;
            k += 10;
            p += 32;
        }
        atomicAdd(&out[cur_batch * width + col], res);
    }
}

#undef BLOCKWIDTH
#undef BLOCKHEIGHT
