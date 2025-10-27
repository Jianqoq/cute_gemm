#include <cuda.h>
#include <stdlib.h>
#include "util.h"

// #define PRINT_INFO
using namespace cute;

template <typename T>
void gen_rand_data(T *data, int n);

template <int a, int b>
constexpr auto div_ceil = (Int<a>{} + Int<b>{} - Int<1>{}) / Int<b>{};

template <typename T, int kTileM, int kTileN, int kTileK, class TiledCopyA, class TiledCopyB, class SLayoutA, class SLayoutB>
__global__ void gemm_copy(T *Cptr, const T *Aptr, const T *Bptr, int m, int n, int k,
                          TiledCopyA copy_a, TiledCopyB copy_b)
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

    auto thr_copy_a = copy_a.get_slice(threadIdx.x);
    auto thr_copy_b = copy_b.get_slice(threadIdx.x);

    Tensor tAgA = thr_copy_a.partition_S(gA); // (num_el_per_cpy, num_copy_m, num_copy_k, k)
    Tensor tBgB = thr_copy_b.partition_S(gB); // (num_el_per_cpy, num_copy_m, num_copy_k, k)
    Tensor tAsA = thr_copy_a.partition_D(sA); // (num_el_per_cpy, num_copy_m, num_copy_k)
    Tensor tBsB = thr_copy_b.partition_D(sB); // (num_el_per_cpy, num_copy_m, num_copy_k)

    copy(copy_a, tAgA(_, _, _, 0), tAsA); // (ACPY,ACPY_M,ACPY_K) -> (ACPY,ACPY_M,ACPY_K)
    copy(copy_b, tBgB(_, _, _, 0), tBsB);
    __syncthreads();

    if (thread0() && blockIdx.x == 0 && blockIdx.y == 0)
    {
        print("tAgA = ");
        print(shape(tAgA(_, _, _, 0)));
        print("\n");
        print("tAsA = ");
        print(shape(tAsA));
        print("\n");
        print("tBgB = ");
        print(shape(tBgB(_, _, _, 0)));
        print("\n");
        print("tBsB = ");
        print(shape(tBsB));
        print("\n");
        print("sA = ");
        print(shape(sA));
        print("\n");
        print("sB = ");
        print(shape(sB));
        print("\n");
        print_tensor(gA);
        print("\n");
        print_tensor(sA);
        print("\n");
        print_layout(sA.layout());
        print("\n");
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
    int m = 32;
    int n = 40;
    int k = 12;

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

    constexpr int kTileM = 4;
    constexpr int kTileN = 5;
    constexpr int kTileK = 6;

    // each thread block handle with (kTileM, kTileN) output
    dim3 grid(n / kTileN, m / kTileM);
    constexpr int block_size = 8;
    dim3 block(block_size);

    using SLayoutA = decltype(make_layout(make_shape(Int<kTileM>{}, Int<kTileK>{}), make_stride(Int<kTileK>{}, Int<1>{})));
    using SLayoutB = decltype(make_layout(make_shape(Int<kTileN>{}, Int<kTileK>{}), make_stride(Int<kTileK>{}, Int<1>{})));

    auto copy_atom_a = Copy_Atom<UniversalCopy<T>, T>{};
    auto copy_atom_b = Copy_Atom<UniversalCopy<T>, T>{};
    auto thread_layout = Layout<Shape<_2, _3>>{};
    auto repeat_layout = Layout<Shape<_2, _1>>{};
    static_assert(size<1>(repeat_layout) % (size(decltype(copy_atom_a)::SrcLayout{}) / (sizeof(T) * 8)) == 0, "each thread must copy a number of copy_atom");
    static_assert(size<1>(repeat_layout) % (size(decltype(copy_atom_b)::SrcLayout{}) / (sizeof(T) * 8)) == 0, "each thread must copy a number of copy_atom");
    static_assert(size(thread_layout) <= block_size, "thread_layout size must be less than or equal to block_size");
    static_assert(size<0>(repeat_layout) * size<0>(thread_layout) <= kTileM, "repeat_layout.0 * thread_layout.0 must be less than or equal to kTileM");
    static_assert(size<1>(repeat_layout) * size<1>(thread_layout) <= kTileN, "repeat_layout.1 * thread_layout.1 must be less than or equal to kTileN");
    // 256线程拷贝 32x8 的输入
    auto copy_a = make_tiled_copy(copy_atom_a, thread_layout, repeat_layout); // copy = (32 * 8, 8)
    auto copy_b = make_tiled_copy(copy_atom_b, thread_layout, repeat_layout); // copy = (32 * 8, 8)
    // print_latex(copy_a);

    int count = 1;
    cudaEventRecord(start);
    for (int i = 0; i < count; ++i)
    {
        gemm_copy<T, kTileM, kTileN, kTileK, decltype(copy_a), decltype(copy_b), SLayoutA, SLayoutB><<<grid, block>>>(Cptr, Aptr, Bptr, m, n, k, copy_a, copy_b);
    }
    auto err = cudaGetLastError();
    printf("err = %d, str = %s\n", err, cudaGetErrorString(err));
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsedTime, start, end);
    std::cout << "gemm-simple took " << elapsedTime / count << "ms." << std::endl;
    // ========== 拷贝GPU结果到host ==========
    cudaMemcpy(Cptr_host_gpu, Cptr, sizeof(T) * m * n, cudaMemcpyDeviceToHost);

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
        data[i] = (float)i;
    }
}