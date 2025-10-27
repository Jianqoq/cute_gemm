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

    for (int k_tile = 0; k_tile < size<2>(gA); ++k_tile)
    {
        copy(copy_a, tAgA(_, _, _, k_tile), tAsA); // (ACPY,ACPY_M,ACPY_K) -> (ACPY,ACPY_M,ACPY_K)
        copy(copy_b, tBgB(_, _, _, k_tile), tBsB);
        cp_async_fence();
        cp_async_wait<0>();
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
void ldmatrixLayout()
{
    TiledMMA mmaC = make_tiled_mma(MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{},
                                   Layout<Shape<_1, _1>>{});
    Copy_Atom<SM75_U32x4_LDSM_N, cute::half_t> s2r_atom_A;

    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_A, mmaC);
    print_latex(s2r_copy_a);
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
    constexpr int block_size = 256;

    // each thread block handle with (kTileM, kTileN) output
    dim3 grid(n / kTileN, m / kTileM);
    dim3 block(block_size);

    using SLayoutA = decltype(make_layout(make_shape(Int<kTileM>{}, Int<kTileK>{}), make_stride(Int<kTileK>{}, Int<1>{})));
    using SLayoutB = decltype(make_layout(make_shape(Int<kTileN>{}, Int<kTileK>{}), make_stride(Int<kTileK>{}, Int<1>{})));

    auto mma = make_tiled_mma(MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{}, Layout<Shape<_1, _2>>{});

    auto copy_atom = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>>, T>{};
    auto thread_layout = Layout<Shape<_32, _4>>{};
    auto repeat_layout = Layout<Shape<_1, _8>>{};
    static_assert(size<1>(repeat_layout) % (size(decltype(copy_atom)::SrcLayout{}) / (sizeof(T) * 8)) == 0, "each thread must copy a number of copy_atom");
    static_assert(size(thread_layout) <= block_size, "thread_layout size must be less than or equal to block_size");
    static_assert(size<0>(repeat_layout) * size<0>(thread_layout) <= kTileM, "repeat_layout.0 * thread_layout.0 must be less than or equal to kTileM");
    static_assert(size<1>(repeat_layout) * size<1>(thread_layout) <= kTileN, "repeat_layout.1 * thread_layout.1 must be less than or equal to kTileN");
    // 256线程拷贝 32x8 的输入
    auto copy_a = make_tiled_copy(copy_atom, thread_layout, repeat_layout);
    auto copy_b = make_tiled_copy(copy_atom, thread_layout, repeat_layout); // copy = (32 * 8, 8)

    // print(copy_b);

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

    // ldmatrixLayout();

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