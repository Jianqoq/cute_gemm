#include <cuda.h>
#include <stdlib.h>
#include "util.h"

// #define PRINT_INFO
using namespace cute;

template <typename T>
void gen_rand_data(T *data, int n);

template <int a, int b>
constexpr auto div_ceil = (Int<a>{} + Int<b>{} - Int<1>{}) / Int<b>{};

template <typename T, int kTileM, int kTileN, int kTileK, class TiledCopyA, class TiledCopyB, class TiledMMA, class SLayoutA, class SLayoutB>
__global__ void gemm_mma(T *Cptr, const T *Aptr, const T *Bptr, int m, int n, int k,
                         TiledCopyA copy_a, TiledCopyB copy_b, TiledMMA mma)
{
    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(Int<1>{}, n));

    int ix = blockIdx.x;
    int iy = blockIdx.y;

    //  gA(kTileM, kTileK, num_tile_k)
    Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
    //  gB(kTileN, kTileK, num_tile_k)
    Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
    //  gC(kTileM, kTileN)
    Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));

    __shared__ T smemA[cosize_v<SLayoutA>];
    __shared__ T smemB[cosize_v<SLayoutB>];

    Tensor sA = make_tensor(make_smem_ptr(smemA), SLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(smemB), SLayoutB{});

    auto thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    Tensor tCsA = thr_mma.partition_A(sA); // (MMA,num_MMA_m,num_MMA_k)
    Tensor tCsB = thr_mma.partition_B(sB); // (MMA,num_MMA_n,num_MMA_k)
    clear(tCrC);

    auto thr_copy_a = copy_a.get_slice(threadIdx.x);
    auto thr_copy_b = copy_b.get_slice(threadIdx.x);

    Tensor tAgA = thr_copy_a.partition_S(gA); // (num_el_per_cpy, num_copy_m, num_copy_k, k)
    Tensor tBgB = thr_copy_b.partition_S(gB); // (num_el_per_cpy, num_copy_m, num_copy_k, k)
    Tensor tAsA = thr_copy_a.partition_D(sA); // (num_el_per_cpy, num_copy_m, num_copy_k)
    Tensor tBsB = thr_copy_b.partition_D(sB); // (num_el_per_cpy, num_copy_m, num_copy_k)

    if (thread0() && blockIdx.x == 0 && blockIdx.y == 0) {
        print("tAgA = ");print(shape(tAgA(_, _, _, 0)));print("\n");
        print("tAsA = ");print(shape(tAsA));print("\n");
        print("tBgB = ");print(shape(tBgB(_, _, _, 0)));print("\n");
        print("tBsB = ");print(shape(tBsB));print("\n");
        print("tCrC = ");print(shape(tCrC));print("\n");
    }
    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K

    CUTE_STATIC_ASSERT_V(  shape(tCrC) ==   shape(tCgC));                // (MMA,MMA_M,MMA_N)
    CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCsA));                // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCsB));                // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                // MMA_K
    for (int k_tile = 0; k_tile < size<2>(gA); ++k_tile)
    {
        copy(copy_a, tAgA(_, _, _, k_tile), tAsA); // (ACPY,ACPY_M,ACPY_K) -> (ACPY,ACPY_M,ACPY_K)
        copy(copy_b, tBgB(_, _, _, k_tile), tBsB);
        __syncthreads();

        gemm(mma, tCsA, tCsB, tCrC);
        __syncthreads();
    }
    copy(tCrC, tCgC);
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
            C[m + n * M] = T(sum);
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
    constexpr int kTileK = 8;

    // each thread block handle with (kTileM, kTileN) output
    dim3 grid(n / kTileN, m / kTileM);
    constexpr int block_size = 256;
    dim3 block(block_size);

    using SLayoutA = decltype(make_layout(make_shape(Int<kTileM>{}, Int<kTileK>{}), make_stride(Int<1>{}, Int<kTileM>{} + Int<1>{})));
    using SLayoutB = decltype(make_layout(make_shape(Int<kTileN>{}, Int<kTileK>{}), make_stride(Int<1>{}, Int<kTileN>{} + Int<1>{})));

    using copy_atom_a = Copy_Atom<UniversalCopy<T>, T>;
    using copy_atom_b = Copy_Atom<UniversalCopy<T>, T>;

    // 256线程拷贝 32x8 的输入, M方向大小必须小于tileM， N方向大小必须小于tileN， K方向大小必须小于tileK
    auto copy_a = make_tiled_copy(copy_atom_a{}, Layout<Shape<_32, _8>>{}, Layout<Shape< _4,_1>>{}); // copy = (32 * 8, 8)
    auto copy_b = make_tiled_copy(copy_atom_b{}, Layout<Shape<_32, _8>>{}, Layout<Shape< _4,_1>>{}); // copy = (32 * 8, 8)
    // print_latex(copy_b);

    using mma_atom = UniversalFMA<T, T, T>;

    // 256线程计算 16x16 的输出
    auto mma = make_tiled_mma(mma_atom{}, Layout<Shape<_16, _16, _1>>{});

    int count = 1;
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