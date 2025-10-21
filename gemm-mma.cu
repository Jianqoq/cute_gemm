#include <cuda.h>
#include <stdlib.h>
#include "util.h"

// #define PRINT_INFO
using namespace cute;

template <typename T>
void gen_rand_data(T *data, int n);

template <typename T, int kTileM, int kTileN, int kTileK, class TiledCopyA, class TiledCopyB, class TiledMMA, class SLayoutA, class SLayoutB>
__global__ void gemm_mma(T *Cptr, const T *Aptr, const T *Bptr, int m, int n, int k,
                         TiledCopyA copy_a, TiledCopyB copy_b, TiledMMA mma)
{
    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

    int ix = blockIdx.x;
    int iy = blockIdx.y;

    //  gA(kTileM, kTileK, num_tile_k)
    //  gB(kTileN, kTileK, num_tile_k)
    //  gC(kTileM, kTileN)
    Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
    Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
    Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));

    __shared__ T smemA[cosize_v<SLayoutA>];
    __shared__ T smemB[cosize_v<SLayoutB>];

    Tensor sA = make_tensor(make_smem_ptr(smemA), SLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(smemB), SLayoutB{});

    auto thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCrC = thr_mma.partition_fragment_C(gC);
    Tensor tCgC = thr_mma.partition_C(gC);
    clear(tCrC);
    // for (int k_tile = 0; k_tile < size<2>(gA); ++k_tile)
    // {
    //     Tensor kgA = gA(_, _, k_tile);
    //     Tensor kgB = gB(_, _, k_tile);
    //     Tensor tAgA = thr_mma.partition_A(kgA);
    //     Tensor tBgB = thr_mma.partition_B(kgB); // 每个线程的 B 部分

    //     Tensor tAsA = local_partition(sA, tA, threadIdx.x);
    //     Tensor tBsB = local_partition(sB, tB, threadIdx.x);
    //     copy(tAgA, tAsA);
    //     copy(tBgB, tBsB);

    //     cp_async_fence();
    //     cp_async_wait<0>();
    //     __syncthreads();

    //     auto tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});
    //     auto tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{});
    //     gemm(tCsA, tCsB, tCrC);

    //     __syncthreads();
    // }
    //   copy(tCrC, tCgC);
}

template <typename T>
void gemm_cpu_reference(const T *A, const T *B, T *C, int M, int N, int K)
{
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                // A 是 (M, K)，B 是 (N, K) 转置布局
                sum += float(A[m * K + k]) * float(B[n * K + k]);
            }
            C[m * N + n] = T(sum);
        }
    }
}

template <typename T>
void verify_result(const T *C_gpu, const T *C_cpu, int size, const char *name)
{
    float max_error = 0.0f;
    float avg_error = 0.0f;
    int error_count = 0;

    for (int i = 0; i < size; ++i)
    {
        float gpu_val = float(C_gpu[i]);
        float cpu_val = float(C_cpu[i]);
        float diff = fabs(gpu_val - cpu_val);
        float rel_error = diff / (fabs(cpu_val) + 1e-5f); // 相对误差

        max_error = fmax(max_error, diff);
        avg_error += diff;

        if (rel_error > 1e-2f)
        { // 相对误差大于1%
            error_count++;
            if (error_count < 10)
            { // 只打印前10个错误
                printf("[%s] Error at %d: GPU=%.6f, CPU=%.6f, diff=%.6f\n",
                       name, i, gpu_val, cpu_val, diff);
            }
        }
    }

    avg_error /= size;

    printf("\n=== %s Verification ===\n", name);
    printf("Max error: %.6f\n", max_error);
    printf("Avg error: %.6f\n", avg_error);
    printf("Error count (>1%%): %d / %d\n", error_count, size);

    if (max_error < 0.1f && error_count < size * 0.01)
    {
        printf("✅ Result PASSED!\n\n");
    }
    else
    {
        printf("❌ Result FAILED!\n\n");
    }
}

int main()
{
    srand(1000);

    using T = cute::half_t;
    cudaEvent_t start, end;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    T *Cptr;
    T *Aptr;
    T *Bptr;
    T *Cptr_host_cpu;
    T *Cptr_host_gpu;
    int m = 128 * 4;
    int n = 128 * 4;
    int k = 128 * 4;

    cudaMalloc(&Cptr, sizeof(T) * m * n);
    cudaMalloc(&Aptr, sizeof(T) * m * k);
    cudaMalloc(&Bptr, sizeof(T) * k * n);

    T *Aptr_host;
    T *Bptr_host;
    Aptr_host = (T *)malloc(sizeof(T) * m * k);
    Bptr_host = (T *)malloc(sizeof(T) * n * k);
    Cptr_host_gpu = (T *)malloc(sizeof(T) * m * n);
    Cptr_host_cpu = (T *)malloc(sizeof(T) * m * n);
    gen_rand_data(Aptr_host, m * k);
    gen_rand_data(Bptr_host, n * k);

    cudaMemcpy(Aptr, Aptr_host, sizeof(T) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(Bptr, Bptr_host, sizeof(T) * n * k, cudaMemcpyHostToDevice);

    constexpr int kTileM = 128;
    constexpr int kTileN = 128;
    constexpr int kTileK = 32;

    // each thread block handle with (kTileM, kTileN) output
    dim3 grid(n / kTileN, m / kTileM);
    constexpr int block_size = 256;
    dim3 block(block_size);

    using SLayoutA = decltype(make_layout(make_shape(Int<kTileM>{}, Int<kTileK>{})));
    using SLayoutB = decltype(make_layout(make_shape(Int<kTileN>{}, Int<kTileK>{})));
    constexpr int k_dimension = size<1>(SLayoutA{}); // 获取第二个维度（K）

    using copy_atom_a = Copy_Atom<UniversalCopy<T>, T>;
    using copy_atom_b = Copy_Atom<UniversalCopy<T>, T>;

    constexpr int warpSize = 32;
    constexpr int num_el_per_cache_line = sizeof(uint128_t) / sizeof(T);
    // 256线程拷贝 32x8 的输入
    auto copy_a = make_tiled_copy(copy_atom_a{}, Layout<Shape<Int<warpSize>, Int<num_el_per_cache_line>>>{}, Layout<Shape<Int<block_size / warpSize>, Int<1>>>{});
    auto copy_b = make_tiled_copy(copy_atom_b{}, Layout<Shape<Int<warpSize>, Int<num_el_per_cache_line>>>{}, Layout<Shape<Int<block_size / warpSize>, Int<1>>>{});
    // print_latex(copy_a);
    // print_latex(copy_b);

    using mma_atom = UniversalFMA<T, T, T>;
    using copy_atom = Copy_Atom<UniversalCopy<T>, T>;

    // 256线程计算 16x16 的输出
    auto mma = make_tiled_mma(mma_atom{}, Layout<Shape<_16, _16, _1>>{});

    int count = 100;
    cudaEventRecord(start);
    for (int i = 0; i < count; ++i)
    {
        gemm_mma<T, kTileM, kTileN, kTileK, decltype(copy_a), decltype(copy_b), decltype(mma), SLayoutA, SLayoutB><<<grid, block>>>(Cptr, Aptr, Bptr, m, n, k, copy_a, copy_b, mma);
    }
    auto err = cudaGetLastError();
    printf("err = %d, str = %s\n", err, cudaGetErrorString(err));
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsedTime, start, end);
    std::cout << "gemm-simple took " << elapsedTime / count << "ms." << std::endl;
    // ========== 拷贝GPU结果到host ==========
    cudaMemcpy(Cptr_host_gpu, Cptr, sizeof(T) * m * n, cudaMemcpyDeviceToHost);

    // ========== CPU 参考计算 ==========
    printf("\nComputing CPU reference...\n");
    gemm_cpu_reference(Aptr_host, Bptr_host, Cptr_host_cpu, m, n, k);

    // ========== 验证结果 ==========
    verify_result(Cptr_host_gpu, Cptr_host_cpu, m * n, "GEMM");

    // ========== 清理 ==========
    free(Aptr_host);
    free(Bptr_host);
    free(Cptr_host_gpu);
    free(Cptr_host_cpu);
    cudaFree(Aptr);
    cudaFree(Bptr);
    cudaFree(Cptr);

    return 0;
}

template <typename T>
void gen_rand_data(T *data, int n)
{
    for (int i = 0; i < n; ++i)
    {
        float v = (rand() % 200 - 100) * 0.01;
        data[i] = v;
    }
}