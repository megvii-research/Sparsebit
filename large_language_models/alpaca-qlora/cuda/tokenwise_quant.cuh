// forked and changed from https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/softmax.cuh

#include <cub/cub.cuh>
#include <math_constants.h>
#include <assert.h>
#include <cuda.h>

#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif // CUDA_VERSION >= 11000

namespace tokenwise_quant
{

    constexpr int kWarpSize = 32;

    template <typename T>
    struct SumOp
    {
        __device__ __forceinline__ T operator()(const T &a, const T &b) const { return a + b; }
    };

    template <typename T>
    struct MaxOp
    {
        __device__ __forceinline__ T operator()(const T &a, const T &b) const { return max(a, b); }
    };

    template <typename T>
    struct MinOp
    {
        __device__ __forceinline__ T operator()(const T &a, const T &b) const { return min(a, b); }
    };

    template <template <typename> class ReductionOp, typename T, int thread_group_width = kWarpSize>
    __inline__ __device__ T WarpAllReduce(T val)
    {
        for (int mask = thread_group_width / 2; mask > 0; mask /= 2)
        {
            val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
        }
        return val;
    }

    template <template <typename> class ReductionOp, typename T, int block_size>
    __inline__ __device__ T BlockAllReduce(T val)
    {
        typedef cub::BlockReduce<T, block_size> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        __shared__ T result_broadcast;
        T result = BlockReduce(temp_storage).Reduce(val, ReductionOp<T>());
        if (threadIdx.x == 0)
        {
            result_broadcast = result;
        }
        __syncthreads();
        return result_broadcast;
    }

    template <typename T>
    __inline__ __device__ T Inf();

    template <>
    __inline__ __device__ float Inf<float>()
    {
        return CUDART_INF_F;
    }

    template <>
    __inline__ __device__ double Inf<double>()
    {
        return CUDART_INF;
    }

    template <typename T>
    __inline__ __device__ T Div(T a, T b);

    template <>
    __inline__ __device__ float Div<float>(float a, float b)
    {
#ifdef OF_SOFTMAX_USE_FAST_MATH
        return __fdividef(a, b);
#else
        return a / b;
#endif
    }

    template <>
    __inline__ __device__ double Div<double>(double a, double b)
    {
        return a / b;
    }

    template <typename T>
    __inline__ __device__ T RemoveZero(T x)
    {
#define EPS 1e-6
        return x < (T)EPS ? 1 : x;
#undef EPS
    }

    template <typename T>
    __inline__ __device__ T GetInvScale(T smin, T smax)
    {
        return RemoveZero(max(Div(smin, (T)-127), Div(smax, (T)127)));
    }

    __inline__ __device__ int8_t Quantize(float val, float inv_scale)
    {
        return __float2int_rn(Div(val, inv_scale)); // auto type convert (4Byte to 1Byte) in return
    }
    __inline__ __device__ int8_t Quantize(half val, half inv_scale)
    {
        return __half2int_rn(Div(val, inv_scale)); // auto type convert (2Byte to 1Byte) in return
    }

    inline cudaError_t GetNumBlocks(int64_t block_size, int64_t max_blocks, int64_t waves,
                                    int *num_blocks)
    {
        int dev;
        {
            cudaError_t err = cudaGetDevice(&dev);
            if (err != cudaSuccess)
            {
                return err;
            }
        }
        int sm_count;
        {
            cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
            if (err != cudaSuccess)
            {
                return err;
            }
        }
        int tpm;
        {
            cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
            if (err != cudaSuccess)
            {
                return err;
            }
        }
        *num_blocks =
            std::max<int>(1, std::min<int64_t>(max_blocks, sm_count * tpm / block_size * waves));
        return cudaSuccess;
    }

    template <typename T>
    struct DefaultComputeType
    {
        using type = T;
    };

    template <>
    struct DefaultComputeType<half>
    {
        using type = float;
    };

#if CUDA_VERSION >= 11000
    template <>
    struct DefaultComputeType<nv_bfloat16>
    {
        using type = float;
    };
#endif // CUDA_VERSION >= 11000

    template <typename T, int N>
    struct GetPackType
    {
        using type = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
    };

    template <typename T, int N>
    using PackType = typename GetPackType<T, N>::type;

    template <typename T, int N>
    union Pack
    {
        static_assert(sizeof(PackType<T, N>) == sizeof(T) * N, "");
        __device__ Pack()
        {
            // do nothing
        }
        PackType<T, N> storage;
        T elem[N];
    };

    template <typename SRC, typename DST>
    struct DirectLoad
    {
        DirectLoad(const SRC *src, int64_t row_size) : src(src), row_size(row_size) {}
        template <int N>
        __device__ void load(DST *dst, int64_t row, int64_t col) const
        {
            Pack<SRC, N> pack;
            const int64_t offset = (row * row_size + col) / N;
            pack.storage = *(reinterpret_cast<const PackType<SRC, N> *>(src) + offset);
#pragma unroll
            for (int i = 0; i < N; ++i)
            {
                dst[i] = static_cast<DST>(pack.elem[i]);
            }
        }
        const SRC *src;
        int64_t row_size;
    };

    template <typename SRC, typename DST>
    struct DirectStore
    {
        DirectStore(DST *dst, int64_t row_size) : dst(dst), row_size(row_size) {}
        template <int N>
        __device__ void store(const SRC *src, int64_t row, int64_t col)
        {
            Pack<DST, N> pack;
            const int64_t offset = (row * row_size + col) / N;
#pragma unroll
            for (int i = 0; i < N; ++i)
            {
                pack.elem[i] = static_cast<DST>(src[i]);
            }
            *(reinterpret_cast<PackType<DST, N> *>(dst) + offset) = pack.storage;
        }
        DST *dst;
        int64_t row_size;
    };

    template <typename LOAD, typename STORE, typename SCALE_STORE, typename ComputeType, int pack_size, int cols_per_thread,
              int thread_group_width, int rows_per_access, bool padding>
    __global__ void QuantWarpImpl(LOAD load, STORE store, SCALE_STORE scale_store, const int64_t rows, const int64_t cols)
    {
        static_assert(cols_per_thread % pack_size == 0, "");
        static_assert(thread_group_width <= kWarpSize, "");
        static_assert(kWarpSize % thread_group_width == 0, "");
        constexpr int num_packs = cols_per_thread / pack_size;
        assert(cols <= cols_per_thread * thread_group_width);
        ComputeType buf[rows_per_access][cols_per_thread];
        int8_t buf_out[rows_per_access][cols_per_thread];
        const int global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
        const int num_global_thread_group = gridDim.x * blockDim.y;
        const int lane_id = threadIdx.x;
        const int64_t step = num_global_thread_group * rows_per_access;
        for (int64_t row = global_thread_group_id * rows_per_access; row < rows; row += step)
        {
            ComputeType thread_max[rows_per_access];
            ComputeType thread_min[rows_per_access];
#pragma unroll
            for (int row_id = 0; row_id < rows_per_access; ++row_id)
            {
                thread_max[row_id] = -Inf<ComputeType>();
                thread_min[row_id] = Inf<ComputeType>();
                ComputeType *row_buf = buf[row_id];
#pragma unroll
                for (int pack_id = 0; pack_id < num_packs; ++pack_id)
                {
                    const int pack_offset = pack_id * pack_size;
                    const int col = (pack_id * thread_group_width + lane_id) * pack_size;
                    if (!padding || col < cols)
                    {
                        load.template load<pack_size>(row_buf + pack_offset, row + row_id, col);
#pragma unroll
                        for (int i = 0; i < pack_size; ++i)
                        {
                            thread_max[row_id] = max(thread_max[row_id], row_buf[pack_offset + i]);
                        }
#pragma unroll
                        for (int i = 0; i < pack_size; ++i)
                        {
                            thread_min[row_id] = min(thread_min[row_id], row_buf[pack_offset + i]);
                        }
                    }
                }
            }

            ComputeType warp_max[rows_per_access];
#pragma unroll
            for (int row_id = 0; row_id < rows_per_access; ++row_id)
            {
                warp_max[row_id] = WarpAllReduce<MaxOp, ComputeType, thread_group_width>(thread_max[row_id]);
            }

            ComputeType warp_min[rows_per_access];
#pragma unroll
            for (int row_id = 0; row_id < rows_per_access; ++row_id)
            {
                warp_min[row_id] = WarpAllReduce<MinOp, ComputeType, thread_group_width>(thread_min[row_id]);
            }
            ComputeType inv_scales[rows_per_access];
#pragma unroll
            for (int row_id = 0; row_id < rows_per_access; ++row_id)
            {
                inv_scales[row_id] = GetInvScale(warp_min[row_id], warp_max[row_id]);
            }
            // FIXME: export scales
#pragma unroll
            for (int row_id = 0; row_id < rows_per_access; ++row_id)
            {
                ComputeType *row_buf = buf[row_id];
                int8_t *row_buf_out = buf_out[row_id];
#pragma unroll
                for (int i = 0; i < cols_per_thread; ++i)
                {
                    row_buf_out[i] = Quantize(row_buf[i], inv_scales[row_id]);
                }
#pragma unroll
                for (int i = 0; i < num_packs; ++i)
                {
                    const int col = (i * thread_group_width + lane_id) * pack_size;
                    if (!padding || col < cols)
                    {
                        store.template store<pack_size>(row_buf_out + i * pack_size, row + row_id, col);
                    }
                }
                scale_store.template store<rows_per_access>(inv_scales, row + row_id, 0);
            }
        }
    }

    template <typename LOAD, typename STORE, typename SCALE_STORE, typename ComputeType, int pack_size, int cols_per_thread,
              int thread_group_width, int rows_per_access, bool padding>
    inline cudaError_t LaunchQuantWarpImpl(cudaStream_t stream, LOAD load, STORE store, SCALE_STORE scale_store,
                                           const int64_t rows, const int64_t cols)
    {
        constexpr int block_size = 128;
        constexpr int waves = 32;
        static_assert(block_size % thread_group_width == 0, "");
        constexpr int thread_groups_per_block = block_size / thread_group_width;
        dim3 block_dim(thread_group_width, thread_groups_per_block);
        const int64_t num_blocks =
            (rows / rows_per_access + thread_groups_per_block - 1) / thread_groups_per_block;
        int grid_dim_x;
        {
            cudaError_t err = GetNumBlocks(block_size, num_blocks, waves, &grid_dim_x);
            if (err != cudaSuccess)
            {
                return err;
            }
        }
        QuantWarpImpl<LOAD, STORE, SCALE_STORE, ComputeType, pack_size, cols_per_thread, thread_group_width,
                      rows_per_access, padding>
            <<<grid_dim_x, block_dim, 0, stream>>>(load, store, scale_store, rows, cols);
        return cudaPeekAtLastError();
    }

    template <typename LOAD, typename STORE, typename SCALE_STORE, typename ComputeType, int pack_size, int cols_per_thread,
              int thread_group_width, int rows_per_access>
    inline cudaError_t DispatchQuantWarpImplPadding(cudaStream_t stream, LOAD load, STORE store, SCALE_STORE scale_store,
                                                    const int64_t rows, const int64_t cols)
    {
        if (cols == cols_per_thread * thread_group_width)
        {
            return LaunchQuantWarpImpl<LOAD, STORE, SCALE_STORE, ComputeType, pack_size, cols_per_thread,
                                       thread_group_width, rows_per_access, false>(
                stream, load, store, scale_store, rows, cols);
        }
        else
        {
            return LaunchQuantWarpImpl<LOAD, STORE, SCALE_STORE, ComputeType, pack_size, cols_per_thread,
                                       thread_group_width, rows_per_access, true>(
                stream, load, store, scale_store, rows, cols);
        }
    }

    template <typename LOAD, typename STORE, typename SCALE_STORE, typename ComputeType, int pack_size>
    typename std::enable_if<pack_size == 1, cudaError_t>::type DispatchQuantWarpImplCols(
        cudaStream_t stream, LOAD load, STORE store, SCALE_STORE scale_store, const int64_t rows, const int64_t cols)
    {
        if (cols <= 0)
        {
            return cudaErrorInvalidValue;
        }
#define DEFINE_ONE_ELIF(thread_group_width)                                                                  \
    else if (cols <= (thread_group_width)*pack_size)                                                         \
    {                                                                                                        \
        if (rows % 2 == 0)                                                                                   \
            return DispatchQuantWarpImplPadding<LOAD, STORE, SCALE_STORE, ComputeType, pack_size, pack_size, \
                                                thread_group_width, 2>(stream, load, store, scale_store,     \
                                                                       rows, cols);                          \
        else                                                                                                 \
            return DispatchQuantWarpImplPadding<LOAD, STORE, SCALE_STORE, ComputeType, pack_size, pack_size, \
                                                thread_group_width, 1>(stream, load, store, scale_store,     \
                                                                       rows, cols);                          \
    }
        DEFINE_ONE_ELIF(1)
        DEFINE_ONE_ELIF(2)
        DEFINE_ONE_ELIF(4)
        DEFINE_ONE_ELIF(8)
        DEFINE_ONE_ELIF(16)
        DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                                      \
    else if (cols <= (col)*kWarpSize)                                                                             \
    {                                                                                                             \
        return DispatchQuantWarpImplPadding<LOAD, STORE, SCALE_STORE, ComputeType, pack_size, col, kWarpSize, 1>( \
            stream, load, store, scale_store, rows, cols);                                                        \
    }
        DEFINE_ONE_ELIF(2)
        DEFINE_ONE_ELIF(3)
        DEFINE_ONE_ELIF(4)
        DEFINE_ONE_ELIF(5)
        DEFINE_ONE_ELIF(6)
        DEFINE_ONE_ELIF(7)
        DEFINE_ONE_ELIF(8)
        DEFINE_ONE_ELIF(9)
        DEFINE_ONE_ELIF(10)
        DEFINE_ONE_ELIF(11)
        DEFINE_ONE_ELIF(12)
        DEFINE_ONE_ELIF(13)
        DEFINE_ONE_ELIF(14)
        DEFINE_ONE_ELIF(15)
        DEFINE_ONE_ELIF(16)
        DEFINE_ONE_ELIF(17)
        DEFINE_ONE_ELIF(18)
        DEFINE_ONE_ELIF(19)
        DEFINE_ONE_ELIF(20)
        DEFINE_ONE_ELIF(21)
        DEFINE_ONE_ELIF(22)
        DEFINE_ONE_ELIF(23)
        DEFINE_ONE_ELIF(24)
        DEFINE_ONE_ELIF(25)
        DEFINE_ONE_ELIF(26)
        DEFINE_ONE_ELIF(27)
        DEFINE_ONE_ELIF(28)
        DEFINE_ONE_ELIF(29)
        DEFINE_ONE_ELIF(30)
        DEFINE_ONE_ELIF(31)
        DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
        else
        {
            return cudaErrorInvalidValue;
        }
    }

    template <typename LOAD, typename STORE, typename SCALE_STORE, typename ComputeType, int pack_size>
    typename std::enable_if<pack_size == 2, cudaError_t>::type DispatchQuantWarpImplCols(
        cudaStream_t stream, LOAD load, STORE store, SCALE_STORE scale_store, const int64_t rows, const int64_t cols)
    {
        if (cols <= 0)
        {
            return cudaErrorInvalidValue;
        }
#define DEFINE_ONE_ELIF(thread_group_width)                                                                                          \
    else if (cols <= (thread_group_width)*pack_size)                                                                                 \
    {                                                                                                                                \
        if (rows % 2 == 0)                                                                                                           \
        {                                                                                                                            \
            return DispatchQuantWarpImplPadding<LOAD, STORE, SCALE_STORE, ComputeType, pack_size, pack_size, thread_group_width, 2>( \
                stream, load, store, scale_store, rows, cols);                                                                       \
        }                                                                                                                            \
        else                                                                                                                         \
        {                                                                                                                            \
            return DispatchQuantWarpImplPadding<LOAD, STORE, SCALE_STORE, ComputeType, pack_size, pack_size, thread_group_width, 1>( \
                stream, load, store, scale_store, rows, cols);                                                                       \
        }                                                                                                                            \
    }
        DEFINE_ONE_ELIF(1)
        DEFINE_ONE_ELIF(2)
        DEFINE_ONE_ELIF(4)
        DEFINE_ONE_ELIF(8)
        DEFINE_ONE_ELIF(16)
        DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                                      \
    else if (cols <= (col)*kWarpSize)                                                                             \
    {                                                                                                             \
        return DispatchQuantWarpImplPadding<LOAD, STORE, SCALE_STORE, ComputeType, pack_size, col, kWarpSize, 1>( \
            stream, load, store, scale_store, rows, cols);                                                        \
    }
        DEFINE_ONE_ELIF(4)
        DEFINE_ONE_ELIF(6)
        DEFINE_ONE_ELIF(8)
        DEFINE_ONE_ELIF(10)
        DEFINE_ONE_ELIF(12)
        DEFINE_ONE_ELIF(14)
        DEFINE_ONE_ELIF(16)
        DEFINE_ONE_ELIF(18)
        DEFINE_ONE_ELIF(20)
        DEFINE_ONE_ELIF(22)
        DEFINE_ONE_ELIF(24)
        DEFINE_ONE_ELIF(26)
        DEFINE_ONE_ELIF(28)
        DEFINE_ONE_ELIF(30)
        DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
        else
        {
            return cudaErrorInvalidValue;
        }
    }

    template <typename LOAD, typename STORE, typename SCALE_STORE, typename ComputeType>
    struct DispatchQuantWarpImplPackSize
    {
        cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, SCALE_STORE scale_store, const int64_t rows,
                               const int64_t cols)
        {
            if (cols % 2 == 0)
            {
                return DispatchQuantWarpImplCols<LOAD, STORE, SCALE_STORE, ComputeType, 2>(stream, load, store, scale_store, rows, cols);
            }
            else
            {
                return DispatchQuantWarpImplCols<LOAD, STORE, SCALE_STORE, ComputeType, 1>(stream, load, store, scale_store, rows, cols);
            }
        }
    };

    template <typename LOAD, typename STORE, typename SCALE_STORE, typename ComputeType>
    inline cudaError_t DispatchQuantWarpImpl(cudaStream_t stream, LOAD load, STORE store, SCALE_STORE scale_store,
                                             const int64_t rows, const int64_t cols)
    {
        return DispatchQuantWarpImplPackSize<LOAD, STORE, SCALE_STORE, ComputeType>()(stream, load, store, scale_store, rows, cols);
    }

    template <typename LOAD, typename STORE, typename SCALE_STORE, typename ComputeType, int pack_size, int block_size>
    __global__ void QuantBlockSMemImpl(LOAD load, STORE store, SCALE_STORE scale_store, const int64_t rows,
                                       const int64_t cols)
    {
        extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
        auto *buf = reinterpret_cast<ComputeType *>(shared_buf);
        const int tid = threadIdx.x;
        assert(cols % pack_size == 0);
        const int num_packs = cols / pack_size;
        for (int64_t row = blockIdx.x; row < rows; row += gridDim.x)
        {
            ComputeType thread_max = -Inf<ComputeType>();
            ComputeType thread_min = Inf<ComputeType>();
            for (int pack_id = tid; pack_id < num_packs; pack_id += block_size)
            {
                ComputeType pack[pack_size];
                load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
                for (int i = 0; i < pack_size; ++i)
                {
                    buf[i * num_packs + pack_id] = pack[i];
                    thread_max = max(thread_max, pack[i]);
                    thread_min = min(thread_min, pack[i]);
                }
            }
            const ComputeType row_max = BlockAllReduce<MaxOp, ComputeType, block_size>(thread_max);
            const ComputeType row_min = BlockAllReduce<MinOp, ComputeType, block_size>(thread_min);
            const ComputeType inv_scale = GetInvScale(row_min, row_max);
            for (int pack_id = tid; pack_id < num_packs; pack_id += block_size)
            {
                int8_t pack[pack_size];
#pragma unroll
                for (int i = 0; i < pack_size; ++i)
                {
                    pack[i] = Quantize(buf[i * num_packs + pack_id], inv_scale);
                }
                store.template store<pack_size>(pack, row, pack_id * pack_size);
            }
            scale_store.template store<1>(&inv_scale, row, 0);
        }
    }

    template <typename LOAD, typename STORE, typename SCALE_STORE, typename ComputeType, int pack_size, int block_size>
    inline cudaError_t LaunchQuantBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store, SCALE_STORE scale_store, int smem,
                                                const int64_t rows, const int64_t cols)
    {
        constexpr int waves = 32;
        int grid_dim_x;
        {
            cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
            if (err != cudaSuccess)
            {
                return err;
            }
        }
        QuantBlockSMemImpl<LOAD, STORE, SCALE_STORE, ComputeType, pack_size, block_size>
            <<<grid_dim_x, block_size, smem, stream>>>(load, store, scale_store, rows, cols);
        return cudaPeekAtLastError();
    }

    template <typename LOAD, typename STORE, typename SCALE_STORE, typename ComputeType, int pack_size>
    inline cudaError_t TryDispatchQuantBlockSMemImplBlockSize(cudaStream_t stream, LOAD load,
                                                              STORE store, SCALE_STORE scale_store,
                                                              const int64_t rows, const int64_t cols, bool *success)
    {
        constexpr int block_size_conf_1 = 128;
        constexpr int block_size_conf_2 = 256;
        constexpr int block_size_conf_3 = 512;
        constexpr int block_size_conf_4 = 1024;
        const size_t smem = cols * sizeof(ComputeType);
        int max_active_blocks_conf_1;
        {
            cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_active_blocks_conf_1,
                QuantBlockSMemImpl<LOAD, STORE, SCALE_STORE, ComputeType, pack_size, block_size_conf_1>,
                block_size_conf_1, smem);
            if (err != cudaSuccess)
            {
                return err;
            }
        }
        if (max_active_blocks_conf_1 <= 0)
        {
            *success = false;
            return cudaSuccess;
        }
        int max_active_blocks_conf_4;
        {
            cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_active_blocks_conf_4,
                QuantBlockSMemImpl<LOAD, STORE, SCALE_STORE, ComputeType, pack_size, block_size_conf_4>,
                block_size_conf_4, smem);
            if (err != cudaSuccess)
            {
                return err;
            }
        }
        if (max_active_blocks_conf_4 == max_active_blocks_conf_1)
        {
            *success = true;
            return LaunchQuantBlockSMemImpl<LOAD, STORE, SCALE_STORE, ComputeType, pack_size, block_size_conf_4>(stream, load, store, scale_store, smem, rows, cols);
        }
        int max_active_blocks_conf_3;
        {
            cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_active_blocks_conf_3,
                QuantBlockSMemImpl<LOAD, STORE, SCALE_STORE, ComputeType, pack_size, block_size_conf_3>,
                block_size_conf_3, smem);
            if (err != cudaSuccess)
            {
                return err;
            }
        }
        if (max_active_blocks_conf_3 == max_active_blocks_conf_1)
        {
            *success = true;
            return LaunchQuantBlockSMemImpl<LOAD, STORE, SCALE_STORE, ComputeType, pack_size, block_size_conf_3>(stream, load, store, scale_store, smem, rows, cols);
        }
        int max_active_blocks_conf_2;
        {
            cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_active_blocks_conf_2,
                QuantBlockSMemImpl<LOAD, STORE, SCALE_STORE, ComputeType, pack_size, block_size_conf_2>,
                block_size_conf_2, smem);
            if (err != cudaSuccess)
            {
                return err;
            }
        }
        if (max_active_blocks_conf_2 == max_active_blocks_conf_1)
        {
            *success = true;
            return LaunchQuantBlockSMemImpl<LOAD, STORE, SCALE_STORE, ComputeType, pack_size, block_size_conf_2>(stream, load, store, scale_store, smem, rows, cols);
        }
        *success = true;
        return LaunchQuantBlockSMemImpl<LOAD, STORE, SCALE_STORE, ComputeType, pack_size, block_size_conf_1>(stream, load, store, scale_store, smem, rows, cols);
    }

    template <typename LOAD, typename STORE, typename SCALE_STORE, typename ComputeType>
    struct TryDispatchQuantBlockSMemImplPackSize
    {
        cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, SCALE_STORE scale_store, const int64_t rows,
                               const int64_t cols, bool *success)
        {
            if (cols % 2 == 0)
            {
                return TryDispatchQuantBlockSMemImplBlockSize<LOAD, STORE, SCALE_STORE, ComputeType, 2>(
                    stream, load, store, scale_store, rows, cols, success);
            }
            else
            {
                return TryDispatchQuantBlockSMemImplBlockSize<LOAD, STORE, SCALE_STORE, ComputeType, 1>(
                    stream, load, store, scale_store, rows, cols, success);
            }
        }
    };

    template <typename LOAD, typename STORE, typename SCALE_STORE, typename ComputeType>
    inline cudaError_t TryDispatchQuantBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store, SCALE_STORE scale_store,
                                                     const int64_t rows, const int64_t cols,
                                                     bool *success)
    {
        return TryDispatchQuantBlockSMemImplPackSize<LOAD, STORE, SCALE_STORE, ComputeType>()(
            stream, load, store, scale_store, rows, cols, success);
    }

    template <typename LOAD, typename STORE, typename SCALE_STORE, typename ComputeType, int pack_size, int block_size>
    __global__ void QuantBlockUncachedImpl(LOAD load, STORE store, SCALE_STORE scale_store, const int64_t rows,
                                           const int64_t cols)
    {
        const int tid = threadIdx.x;
        assert(cols % pack_size == 0);
        const int num_packs = cols / pack_size;
        for (int64_t row = blockIdx.x; row < rows; row += gridDim.x)
        {
            ComputeType thread_max = -Inf<ComputeType>();
            ComputeType thread_min = Inf<ComputeType>();
            for (int pack_id = tid; pack_id < num_packs; pack_id += block_size)
            {
                ComputeType pack[pack_size];
                load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
                for (int i = 0; i < pack_size; ++i)
                {
                    thread_max = max(thread_max, pack[i]);
                    thread_min = min(thread_min, pack[i]);
                }
            }
            const ComputeType row_max = BlockAllReduce<MaxOp, ComputeType, block_size>(thread_max);
            const ComputeType row_min = BlockAllReduce<MinOp, ComputeType, block_size>(thread_min);
            const ComputeType inv_scale = GetInvScale(row_min, row_max);
            for (int pack_id = tid; pack_id < num_packs; pack_id += block_size)
            {
                ComputeType pack[pack_size];
                int8_t out_pack[pack_size];
                load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
                for (int i = 0; i < pack_size; ++i)
                    out_pack[i] = Quantize(pack[i], inv_scale);
                store.template store<pack_size>(out_pack, row, pack_id * pack_size);
            }
            scale_store.template store<1>(&inv_scale, row, 0);
        }
    }

    template <typename LOAD, typename STORE, typename SCALE_STORE, typename ComputeType, int pack_size>
    inline cudaError_t LaunchQuantBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store, SCALE_STORE scale_store,
                                                    const int64_t rows, const int64_t cols)
    {
        constexpr int block_size = 1024;
        constexpr int waves = 32;
        int grid_dim_x;
        {
            cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
            if (err != cudaSuccess)
            {
                return err;
            }
        }
        QuantBlockUncachedImpl<LOAD, STORE, SCALE_STORE, ComputeType, pack_size, block_size>
            <<<grid_dim_x, block_size, 0, stream>>>(load, store, scale_store, rows, cols);
        return cudaPeekAtLastError();
    }

    template <typename LOAD, typename STORE, typename SCALE_STORE, typename ComputeType>
    struct DispatchQuantBlockUncachedImplPackSize
    {
        cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, SCALE_STORE scale_store, const int64_t rows,
                               const int64_t cols)
        {
            if (cols % 2 == 0)
            {
                return LaunchQuantBlockUncachedImpl<LOAD, STORE, SCALE_STORE, ComputeType, 2>(
                    stream, load, store, scale_store, rows, cols);
            }
            else
            {
                return LaunchQuantBlockUncachedImpl<LOAD, STORE, SCALE_STORE, ComputeType, 1>(
                    stream, load, store, scale_store, rows, cols);
            }
        }
    };

    template <typename LOAD, typename STORE, typename SCALE_STORE, typename ComputeType>
    inline cudaError_t DispatchQuantBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store, SCALE_STORE scale_store,
                                                      const int64_t rows, const int64_t cols)
    {
        return DispatchQuantBlockUncachedImplPackSize<LOAD, STORE, SCALE_STORE, ComputeType>()(
            stream, load, store, scale_store, rows, cols);
    }

    template <typename LOAD, typename STORE, typename SCALE_STORE, typename ComputeType>
    inline typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type
    DispatchQuant(cudaStream_t stream, LOAD load, STORE store, SCALE_STORE scale_store, const int64_t rows, const int64_t cols)
    {
        if (cols < 1024)
        {
            return DispatchQuantWarpImpl<LOAD, STORE, SCALE_STORE, ComputeType>(
                stream, load, store, scale_store, rows, cols);
        }
        else
        {
            bool dispatch_smem_impl_success;
            {
                cudaError_t err =
                    TryDispatchQuantBlockSMemImpl<LOAD, STORE, SCALE_STORE, ComputeType>(
                        stream, load, store, scale_store, rows, cols, &dispatch_smem_impl_success);
                if (err != cudaSuccess)
                {
                    return err;
                }
            }
            if (!dispatch_smem_impl_success)
            {
                return DispatchQuantBlockUncachedImpl<LOAD, STORE, SCALE_STORE, ComputeType>(
                    stream, load, store, scale_store, rows, cols);
            }
            return cudaSuccess;
        }
    }

    template <typename LOAD, typename STORE, typename SCALE_STORE, typename ComputeType>
    inline typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type
    DispatchQuant(cudaStream_t stream, LOAD load, STORE store, SCALE_STORE scale_store,
                  const int64_t rows, const int64_t cols)
    {
        return DispatchQuantBlockUncachedImpl<LOAD, STORE, SCALE_STORE, ComputeType>(
            stream, load, store, scale_store, rows, cols);
    }

} // namespace tokenwise_quant