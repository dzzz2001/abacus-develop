#include "gemm_tn_vbatch.cuh"
#include "gemm_nn_vbatch.cuh"
#include "dgemm_vbatch.h"
#include "source_base/module_device/device.h"

// ----------------------------------------------------------------------------
// Shape-exact dispatch
// ----------------------------------------------------------------------------
//
// The caller (phi_operator_gpu.cu) buckets atom pairs by (nw1, nw2) so every
// item in a batch has exactly the same (m, n, k). The scalars passed here are
// the *exact* per-matrix shapes (not upper bounds), which lets the tile
// ladder pick the tightest template and sizes the grid tightly (no
// over-launched blocks that short-circuit inside the kernel).
//
// Kernel-level dimension mapping (after the A/B swap inside
// vbatched_gemm_*_impl):
//
//   call    | wrapper m    | wrapper n   | wrapper k
//   --------|--------------|-------------|---------------
//   NN      | bxyz (large) | nw2 (small) | nw1 (small)
//   TN      | nw1 (small)  | nw2 (small) | bxyz (large)
//
// (m, n, k) flow through as scalars all the way down into the kernel, so
// there is no per-batchid M/N/K load and no fill-kernel scratch buffer.
// ----------------------------------------------------------------------------

template<typename T>
void gemm_nn_vbatch(
    int m, int n, int k,
    const T* const* A_array_d, const int* lda_d,
    const T* const* B_array_d, const int* ldb_d,
    T** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream,
    const T* alpha)
{
    // 3x2 ladder (6 instantiations), tuned for A100:
    //   n (nw2 axis) -> BLK_M in {8, 16, 32}
    //   m (bxyz axis) -> BLK_N in {32, 64}      (capped at 64)
    //   BLK_K fixed at 16                        (nw1 axis)
    //
    // BLK_N is capped at 64 because A100's 108 SMs benefit more from a larger
    // block count than from a single oversized tile per matrix. bxyz > 64
    // wraps into ceil(bxyz/64) N-tiles; total flop waste in the tail tile is
    // identical to a BLK_N=128 single-tile layout, but SM occupancy roughly
    // doubles (~11 KB shmem vs ~21 KB, ~100 regs/thread vs ~150), so latency
    // hiding improves.
    //
    // Block shape is DIM_X=8 x DIM_Y=16 (128 threads). Every (BLK_M, BLK_N)
    // pair satisfies BLK_M % DIM_X == 0 and BLK_N % DIM_Y == 0, so all six
    // combinations compile to valid kernels. Boundary tiles that exceed the
    // per-matrix M/N are masked out by the in-kernel store guard (harmless
    // here since exact shapes leave at most one boundary tile per axis).
    #define NN_DISPATCH(BLK_M_, BLK_N_)                                    \
        vbatched_gemm_nn_impl<T, 8, 16, BLK_M_, BLK_N_, 16, 8, 16, 8, 16>( \
            m, n, k,                                                       \
            A_array_d, lda_d, B_array_d, ldb_d,                            \
            C_array_d, ldc_d, batchCount, stream, alpha)

    if (n <= 8) {
        if (m <= 32) { NN_DISPATCH( 8, 32); }
        else         { NN_DISPATCH( 8, 64); }
    } else if (n <= 16) {
        if (m <= 32) { NN_DISPATCH(16, 32); }
        else         { NN_DISPATCH(16, 64); }
    } else {
        if (m <= 32) { NN_DISPATCH(32, 32); }
        else         { NN_DISPATCH(32, 64); }
    }

    #undef NN_DISPATCH
}

template<typename T>
void gemm_tn_vbatch(
    int m, int n, int k,
    const T* const* A_array_d, const int* lda_d,
    const T* const* B_array_d, const int* ldb_d,
    T** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream,
    const T* alpha)
{
    // 3x3 ladder (9 instantiations), tuned for A100:
    //   n (nw2 axis) -> BLK_M in {8, 16, 32}
    //   m (nw1 axis) -> BLK_N in {8, 16, 32}
    //   BLK_K fixed at 32                        (bxyz axis)
    //
    // BLK_K is not split by bxyz: the K-axis tail wastes only shmem loads
    // (not FMAs), so a single BLK_K keeps the template table small while
    // still covering bxyz in [27, 125] via ceil(bxyz/32) K-tiles. bxyz=27
    // fits in one tile (5/32 = 16% load waste); larger bxyz wraps into
    // 2-4 K-tiles with modest __syncthreads() overhead.
    //
    // Block shape is DIM_X=8 x DIM_Y=8 (64 threads). Every (BLK_M, BLK_N)
    // pair is divisible by DIM_X/DIM_Y/DIM_XA/DIM_YA/DIM_XB/DIM_YB = 8,
    // so all nine combinations compile to valid kernels.
    #define TN_DISPATCH(BLK_M_, BLK_N_)                                 \
        vbatched_gemm_tn_impl<T, 8, 8, BLK_M_, BLK_N_, 32, 8, 8, 8, 8>( \
            m, n, k,                                                    \
            A_array_d, lda_d, B_array_d, ldb_d,                         \
            C_array_d, ldc_d, batchCount, stream, alpha)

    if (n <= 8) {
        if      (m <=  8) { TN_DISPATCH( 8,  8); }
        else if (m <= 16) { TN_DISPATCH( 8, 16); }
        else              { TN_DISPATCH( 8, 32); }
    } else if (n <= 16) {
        if      (m <=  8) { TN_DISPATCH(16,  8); }
        else if (m <= 16) { TN_DISPATCH(16, 16); }
        else              { TN_DISPATCH(16, 32); }
    } else {
        if      (m <=  8) { TN_DISPATCH(32,  8); }
        else if (m <= 16) { TN_DISPATCH(32, 16); }
        else              { TN_DISPATCH(32, 32); }
    }

    #undef TN_DISPATCH
}

// Explicit instantiations
template void gemm_nn_vbatch<double>(
    int, int, int,
    const double* const*, const int*, const double* const*, const int*,
    double**, const int*, int, cudaStream_t, const double*);

template void gemm_nn_vbatch<float>(
    int, int, int,
    const float* const*, const int*, const float* const*, const int*,
    float**, const int*, int, cudaStream_t, const float*);

template void gemm_tn_vbatch<double>(
    int, int, int,
    const double* const*, const int*, const double* const*, const int*,
    double**, const int*, int, cudaStream_t, const double*);

template void gemm_tn_vbatch<float>(
    int, int, int,
    const float* const*, const int*, const float* const*, const int*,
    float**, const int*, int, cudaStream_t, const float*);
