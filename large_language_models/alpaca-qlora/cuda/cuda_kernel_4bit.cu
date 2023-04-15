#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>

using at::cuda::getCurrentCUDAStream;

#if TORCH_VERSION_MAJOR <= 1 && TORCH_VERSION_MINOR <= 10 // torch version <= 1.10, use local file
#include "Atomic.cuh"
#else
#include <ATen/cuda/Atomic.cuh>
#endif

template <typename scalar_t>
__global__ void VecQuant4MatMulKernel(
    const scalar_t *__restrict__ inp1,
    const int *__restrict__ inp2,
    scalar_t *__restrict__ out,
    const scalar_t *__restrict__ scales,
    const scalar_t *__restrict__ zeros,
    int height,
    int width,
    int inchannels,
    int batch,
    int group_size);

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVUP(a, b) (((a) + (b)-1) / (b))
#define BIT 4
#define BLOCKWIDTH_4BIT 64
#define BLOCKHEIGHT_4BIT 16
#define BLOCKLEN_4BIT (BLOCKHEIGHT_4BIT * 32 / BIT)
#define READ_REPEAT_NUM (BLOCKLEN_4BIT / BLOCKWIDTH_4BIT)

#define BLOCKWIDTH BLOCKWIDTH_4BIT
#define BLOCKHEIGHT BLOCKHEIGHT_4BIT
#define BLOCKLEN BLOCKLEN_4BIT

#define DIM_LEN(x) ((x).sizes().vec().size())

void vecquant4matmul_cuda(
    torch::Tensor inp1,
    torch::Tensor inp2,
    torch::Tensor out,
    torch::Tensor scales,
    torch::Tensor zeros,
    int group_size = 0)
{
    TORCH_CHECK(DIM_LEN(inp1) >= 2, "input1 must be with dimension >= 2");
    int inchannels = inp1.size(-1);
    int batch = inp1.numel() / inchannels;

    TORCH_CHECK(DIM_LEN(inp2) == 2, "input2 must be with dimension == 2");
    int height = inp2.size(0);
    int width = inp2.size(1);

    TORCH_CHECK(out.size(-1) == width, "output channel must be the same with input2 out_channel");
    dim3 blocks(
        (height + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
        (width + BLOCKWIDTH - 1) / BLOCKWIDTH);
    dim3 threads(BLOCKWIDTH);

    if (group_size != 0)
    {
        TORCH_CHECK(group_size / 128 * 128 == group_size, "only group_size divisible by 128 is supported in 4-bit quantization");
    }
    else
        group_size = inchannels;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        inp1.type(), "vecquant4matmul_cuda",
        ([&]
         { VecQuant4MatMulKernel<<<blocks, threads, BLOCKHEIGHT * BLOCKWIDTH * sizeof(unsigned) + BLOCKLEN * 2 * sizeof(scalar_t), stream>>>(
               inp1.data<scalar_t>(),
               inp2.data<int>(),
               out.data<scalar_t>(),
               scales.data<scalar_t>(),
               zeros.data<scalar_t>(),
               height,
               width,
               inchannels,
               batch,
               group_size); }));
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
    int inchannels,
    int batch,
    int group_size)
{
    __shared__ unsigned int weight[BLOCKHEIGHT * BLOCKWIDTH];
    __shared__ scalar_t inputs[BLOCKLEN << 1];

    int row = BLOCKHEIGHT * blockIdx.x;
    int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
    int inp1_pos = BLOCKLEN * blockIdx.x + threadIdx.x;

    int cur = 0, pre = BLOCKLEN;
    auto cur_input = inputs + cur, pre_input = inputs + pre;
    int group_max = DIVUP(inchannels, group_size);
    int group_pos = row * 32 / 4 / group_size;
    scalar_t scale = col < width ? scales[group_max * col + group_pos] : (scalar_t)0;
    scalar_t zero = col < width ? zeros[group_max * col + group_pos] : (scalar_t)0;
    scalar_t res;

    int block_height = col < width ? MIN(BLOCKHEIGHT, height - row) : 0;
    int block_height8 = block_height * 8;
    int read_times = MIN(READ_REPEAT_NUM, (inchannels - inp1_pos + BLOCKWIDTH - 1) / BLOCKWIDTH);

    for (int i = width * row + col, j = 0; j < block_height; j += 1, i += width)
        weight[j * BLOCKWIDTH + threadIdx.x] = as_unsigned(inp2[i]);

    for (int i = 0; i < READ_REPEAT_NUM; ++i)
        if (i < read_times)
            pre_input[i * BLOCKWIDTH + threadIdx.x] = inp1[inp1_pos + i * BLOCKWIDTH];
        else
        {
            pre_input[i * BLOCKWIDTH + threadIdx.x] = 0;
            cur_input[i * BLOCKWIDTH + threadIdx.x] = 0;
        }

    __syncthreads();
    int output_pos = col;
    for (int b = 1; b < batch; ++b)
    {
        cur = BLOCKLEN - cur;
        pre = BLOCKLEN - pre;
        cur_input = inputs + cur;
        pre_input = inputs + pre;
        // extract b-th data
        for (int i = 0; i < read_times; ++i)
            pre_input[i * BLOCKWIDTH + threadIdx.x] = inp1[b * inchannels + i * BLOCKWIDTH + inp1_pos];

        // (b - 1) - th calculation
        res = 0;
        for (int k = 0, i = threadIdx.x; k < block_height8; k += 8, i += BLOCKWIDTH){
            for (int shift=0; shift < 8; shift++){
                res += (scale * scalar_t((weight[i] >> (4*shift)) & 0xF) - zero) * cur_input[k | shift];
            }
        }
        if (col < width)
            gpuAtomicAdd(out + output_pos, res);
        output_pos += width;
        __syncthreads();
    }
    // do batch - th calculation

    cur_input = inputs + pre;
    res = 0;
    for (int k = 0, i = threadIdx.x; k < block_height8; k += 8, i += BLOCKWIDTH){
        for (int shift=0; shift < 8; shift++){
            res += (scale * scalar_t((weight[i] >> (4*shift)) & 0xF) - zero) * cur_input[k | shift];
        }
    }

    if (col < width)
        gpuAtomicAdd(out + output_pos, res);
}

#define BLOCKWIDTH_B 256
#define BLOCKHEIGHT_B 16
#define BLOCKSTRIDE (BLOCKHEIGHT * 32 / BIT)

template <typename scalar_t>
__global__ void VecQuant4MatMulBackwardKernel(
    const scalar_t *__restrict__ grad,
    const int *__restrict__ w,
    scalar_t *__restrict__ out,
    const scalar_t *__restrict__ scales,
    const scalar_t *__restrict__ zeros,
    int height,
    int width,
    int inchannels,
    int group_size);

void vecquant4matmulbackward_cuda(
    torch::Tensor grad,
    torch::Tensor w,
    torch::Tensor out,
    torch::Tensor scales,
    torch::Tensor zeros,
    int group_size = 0)
{
    TORCH_CHECK(DIM_LEN(grad) >= 2, "grad must be with dimension >= 2");
    int outchannels = grad.size(-1);
    int batch = grad.numel() / outchannels;

    TORCH_CHECK(DIM_LEN(w) == 2, "weight must be with dimension == 2");
    int height = w.size(0);
    int width = w.size(1);
    int inchannels = out.size(-1);

    dim3 blocks(batch, DIVUP(height, BLOCKHEIGHT_B), DIVUP(width, BLOCKWIDTH_B));
    dim3 threads(BLOCKHEIGHT_B * 32 / BIT);

    if (group_size != 0)
    {
        TORCH_CHECK(group_size / 128 * 128 == group_size, "only group_size divisible by 128 is supported in 4-bit quantization");
    }
    else
        group_size = inchannels;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(
        grad.type(), "vecquant4matmulbackward_cuda",
        ([&]
         { VecQuant4MatMulBackwardKernel<<<blocks, threads, 0, stream>>>(
               grad.data<scalar_t>(),
               w.data<int>(),
               out.data<scalar_t>(),
               scales.data<scalar_t>(),
               zeros.data<scalar_t>(),
               height,
               width,
               inchannels,
               group_size); }));
}

template <typename scalar_t>
__global__ void VecQuant4MatMulBackwardKernel(
    const scalar_t *__restrict__ grad, // (N, OC)
    const int *__restrict__ w,         // (IC, OC)
    scalar_t *__restrict__ out,        // (N, IC)
    const scalar_t *__restrict__ scales,
    const scalar_t *__restrict__ zeros,
    int height,
    int width,
    int inchannels,
    int group_size)
{
    __shared__ unsigned weights[BLOCKHEIGHT_B * BLOCKWIDTH_B];
    __shared__ scalar_t scale[BLOCKWIDTH_B], zero[BLOCKWIDTH_B], inp[BLOCKWIDTH_B];
    int cur_batch = blockIdx.x;
    int row = BLOCKHEIGHT_B * blockIdx.y;
    int col = BLOCKWIDTH_B * blockIdx.z;
    int line_pos = row * 32 / BIT + threadIdx.x;
    int group_pos = row * 32 / BIT / group_size;
    int group_max = DIVUP(inchannels, group_size);

    for (int i = threadIdx.x; i < BLOCKWIDTH_B; i += BLOCKSTRIDE)
        if (i < width)
        {
            scale[i] = scales[group_max * (col + i) + group_pos];
            zero[i] = zeros[group_max * (col + i) + group_pos];
            inp[i] = grad[cur_batch * width + col + i];
        }
        else
        {
            scale[i] = 0;
            zero[i] = 0;
            inp[i] = 0;
        }

    for (int i = threadIdx.x; i < BLOCKWIDTH_B && col + i < width; i += BLOCKSTRIDE)
    {
        for (int j = 0; j < BLOCKHEIGHT_B && row + j < height; ++j)
            weights[j * BLOCKWIDTH_B + i] = as_unsigned(w[(row + j) * width + col + i]);
    }
    __syncthreads();
    scalar_t res = 0;
    if (line_pos < inchannels)
    {
        int wpos = (threadIdx.x * BIT / 32) * BLOCKWIDTH_B;
        int wshift = (threadIdx.x * BIT) & 0x1F;
        for (int i = 0; i < BLOCKWIDTH_B && col + i < width; ++i)
            res += (scale[i] * scalar_t((weights[wpos + i] >> wshift) & 0xF) - zero[i]) * inp[i];
        gpuAtomicAdd(&out[cur_batch * inchannels + line_pos], res);
    }
}


#undef BLOCKWIDTH
#undef BLOCKHEIGHT
#undef BLOCKLEN
#undef DIM_LEN
