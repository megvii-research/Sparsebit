#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void VecQuant3MatMulKernel(
    const scalar_t *__restrict__ vec,
    const int *__restrict__ mat,
    scalar_t *__restrict__ mul,
    const scalar_t *__restrict__ scales,
    const scalar_t *__restrict__ zeros,
    int height,
    int width,
    int inchannels);

const int BLOCKWIDTH_3BIT = 1024;
const int BLOCKHEIGHT_3BIT = 96;

#define BLOCKWIDTH BLOCKWIDTH_3BIT
#define BLOCKHEIGHT BLOCKHEIGHT_3BIT

void vecquant3matmul_cuda(
    torch::Tensor vec,
    torch::Tensor mat,
    torch::Tensor mul,
    torch::Tensor scales,
    torch::Tensor zeros)
{
    int height = mat.size(0);
    int width = mat.size(1);
    int inchannels = vec.size(-1);

    dim3 blocks(
        (height + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
        (width + BLOCKWIDTH - 1) / BLOCKWIDTH);
    dim3 threads(BLOCKWIDTH);
    AT_DISPATCH_FLOATING_TYPES(
        vec.type(), "vecquant3matmul_cuda",
        ([&]
         { VecQuant3MatMulKernel<<<blocks, threads>>>(
               vec.data<scalar_t>(),
               mat.data<int>(),
               mul.data<scalar_t>(),
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
__global__ void VecQuant3MatMulKernel(
    const scalar_t *__restrict__ vec,
    const int *__restrict__ mat,
    scalar_t *__restrict__ mul,
    const scalar_t *__restrict__ scales,
    const scalar_t *__restrict__ zeros,
    int height,
    int width,
    int inchannels)
{
    int row = BLOCKHEIGHT * blockIdx.x;
    int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
    int vec_pos = BLOCKWIDTH * blockIdx.x + threadIdx.x;

    __shared__ scalar_t blockvec[BLOCKWIDTH];
    if (vec_pos < inchannels)
        blockvec[threadIdx.x] = vec[vec_pos];
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

        unsigned int tmp1;
        unsigned int tmp2;
        unsigned int tmp;

        while (k < BLOCKWIDTH && i < max_i)
        {
            tmp1 = as_unsigned(mat[i]);
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
                tmp2 = as_unsigned(mat[i]);
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
                tmp1 = as_unsigned(mat[i]);
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
        }

        atomicAdd(&mul[col], res);
    }
}

#undef BLOCKWIDTH
#undef BLOCKHEIGHT
