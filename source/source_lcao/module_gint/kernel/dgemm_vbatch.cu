#include "gemm_tn_vbatch.cuh"
#include "gemm_nn_vbatch.cuh"
#include "dgemm_vbatch.h"
#include "source_base/module_device/device.h"

// ----------------------------------------------------------------------------
// Shape-based dispatch
// ----------------------------------------------------------------------------
//
// The two wrappers below select a kernel template instantiation from
// (max_m, max_n, max_k) of the sub-batch instead of an externally-supplied
// bucket id. Callers (`phi_mul_phi`, `phi_mul_dm` in phi_operator_gpu.cu) are
// still expected to partition their full batch into shape-homogeneous
// sub-batches before calling -- this wrapper just routes to the right tile.
//
// Recall the kernel-level dimension mapping after the A/B swap inside
// `vbatched_gemm_{nn,tn}_impl`:
//
//   call    | wrapper max_m | wrapper max_n | wrapper max_k
//   --------|---------------|---------------|---------------
//   NN      | bxyz (const)  | nw2           | nw1
//   TN      | nw1           | nw2           | bxyz (const)
//
// The dispatch boundaries below are chosen so that on the current production
// call sites (cases bxyz <= 64), the wrapper picks exactly the same template
// the previous bucket-id-based dispatch would have. The third tier targets
// large bxyz (>= 80) and is additive -- it gives future tuning a landing pad
// without disturbing today's call sites.
// ----------------------------------------------------------------------------

template<typename T>
void gemm_nn_vbatch(
    int max_m, int max_n, int max_k,
    const int* m_d, const int* n_d, const int* k_d,
    const T* const* A_array_d, const int* lda_d,
    const T* const* B_array_d, const int* ldb_d,
    T** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream,
    const T* alpha)
{
    // NN dimension mapping (after A/B swap in _impl):
    //   kernel M = n (nw2, small 2-27), kernel N = m (bxyz, large 27-125),
    //   kernel K = k (nw1, small 2-27).
    // Tile shape: small BLK_M x large BLK_N x small BLK_K.
    //
    // Dispatch key:
    //   max_n  -> selects BLK_M (nw2 axis)
    //   max_m  -> selects BLK_N (bxyz axis)
    //
    // Tier 0 (max_n <= 8): small BLK_M=8 -- avoids wasting threads on the
    //                      nw2 axis when nw2 is small (Li/O/B/C/N/F-only
    //                      sub-batches).
    // Tier 1 (max_n > 8 && max_m <= 64):
    //                      BLK_M=16, BLK_N=64. Covers bxyz in {27, 48, 64}
    //                      with BLK_N=64 in a single N-tile, and handles
    //                      nw2 in [9, 27] with two BLK_M=16 tiles at full
    //                      thread density (16+11 for nw2=27, 100% useful).
    //                      This is today's "bucket 1+2" tile and the
    //                      production code path for cases 1-3.
    // Tier 2 (max_n > 8 && max_m  > 64):
    //                      BLK_M=16, BLK_N=128. New tier for bxyz in
    //                      {80, 100, 125}. BLK_N=128 covers bxyz=125 in a
    //                      single N-tile (was 2 tiles at BLK_N=64). Same
    //                      BLK_M / BLK_K as tier 1 so behavior degrades
    //                      gracefully if a benchmark turns up unexpected
    //                      regressions; only the bxyz axis changes.
    if (max_n <= 8) {
        //                         DIM_X,Y  BLK_M,N,K   DIM_XA,YA  DIM_XB,YB
        vbatched_gemm_nn_impl<T,   8, 16,    8,  64, 16, 8, 16,    8, 16>
            (max_m, max_n, m_d, n_d, k_d,
             A_array_d, lda_d, B_array_d, ldb_d,
             C_array_d, ldc_d, batchCount, stream, alpha);
    } else if (max_m <= 64) {
        vbatched_gemm_nn_impl<T,   8, 16,   16,  64, 16, 8, 16,    8, 16>
            (max_m, max_n, m_d, n_d, k_d,
             A_array_d, lda_d, B_array_d, ldb_d,
             C_array_d, ldc_d, batchCount, stream, alpha);
    } else {
        vbatched_gemm_nn_impl<T,   8, 16,   16, 128, 16, 8, 16,    8, 16>
            (max_m, max_n, m_d, n_d, k_d,
             A_array_d, lda_d, B_array_d, ldb_d,
             C_array_d, ldc_d, batchCount, stream, alpha);
    }
}

template<typename T>
void gemm_tn_vbatch(
    int max_m, int max_n, int max_k,
    const int* m_d, const int* n_d, const int* k_d,
    const T* const* A_array_d, const int* lda_d,
    const T* const* B_array_d, const int* ldb_d,
    T** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream,
    const T* alpha)
{
    // TN dimension mapping (after A/B swap in _impl):
    //   kernel M = n (nw2, small 2-27), kernel N = m (nw1, small 2-27),
    //   kernel K = k (bxyz, large 27-125).
    // Tile shape: small BLK_M x small BLK_N x large BLK_K.
    //
    // Dispatch key:
    //   max(max_m, max_n) -> selects BLK_M / BLK_N (the nw axes)
    //   max_k             -> selects BLK_K        (the bxyz axis)
    //
    // Tier 0 (max(max_m, max_n) <= 8):
    //                      BLK 8x8x32. Small-nw-only sub-batches.
    // Tier 1 (max(max_m, max_n) > 8 && max_k <= 64):
    //                      BLK 16x16x32. Today's "bucket 1+2" tile and the
    //                      production path for cases 1-3 (bxyz in {27,48,64}).
    //                      BLK_M=BLK_N=16 is intentionally narrow: see the
    //                      asymmetric-pair note below.
    // Tier 2 (max(max_m, max_n) > 8 && max_k  > 64):
    //                      BLK 16x16x64. New tier for bxyz in {80,100,125}.
    //                      Doubled BLK_K halves the K-iter count at bxyz=125
    //                      vs tier 1. BLK_M/BLK_N stay at 16 -- crucially,
    //                      we do not widen the nw-axis tile, so asymmetric
    //                      pairs (TM-O = 13x27, TM-Li = 7x27) keep their
    //                      thread density.
    //
    // Asymmetric-pair safety: previous tuning notes (commit history of this
    // file) document that widening BLK_M / BLK_N beyond 16 wastes ~60% of
    // threads on TM-O (13x27) pairs because the smaller dim only fills part
    // of the tile, and that this dominates the K-iter savings of a wider
    // tile. The new tier therefore only varies BLK_K.
    const int max_mn = max_m > max_n ? max_m : max_n;
    if (max_mn <= 8) {
        //                         DIM_X,Y  BLK_M,N,K   DIM_XA,YA  DIM_XB,YB
        vbatched_gemm_tn_impl<T,   8, 8,     8,  8, 32, 8, 8,      8, 8>
            (max_m, max_n, m_d, n_d, k_d,
             A_array_d, lda_d, B_array_d, ldb_d,
             C_array_d, ldc_d, batchCount, stream, alpha);
    } else if (max_k <= 64) {
        vbatched_gemm_tn_impl<T,   8, 8,    16, 16, 32, 8, 8,      8, 8>
            (max_m, max_n, m_d, n_d, k_d,
             A_array_d, lda_d, B_array_d, ldb_d,
             C_array_d, ldc_d, batchCount, stream, alpha);
    } else {
        vbatched_gemm_tn_impl<T,   8, 8,    16, 16, 64, 8, 8,      8, 8>
            (max_m, max_n, m_d, n_d, k_d,
             A_array_d, lda_d, B_array_d, ldb_d,
             C_array_d, ldc_d, batchCount, stream, alpha);
    }
}

// Explicit instantiations
template void gemm_nn_vbatch<double>(
    int, int, int, const int*, const int*, const int*,
    const double* const*, const int*, const double* const*, const int*,
    double**, const int*, int, cudaStream_t, const double*);

template void gemm_nn_vbatch<float>(
    int, int, int, const int*, const int*, const int*,
    const float* const*, const int*, const float* const*, const int*,
    float**, const int*, int, cudaStream_t, const float*);

template void gemm_tn_vbatch<double>(
    int, int, int, const int*, const int*, const int*,
    const double* const*, const int*, const double* const*, const int*,
    double**, const int*, int, cudaStream_t, const double*);

template void gemm_tn_vbatch<float>(
    int, int, int, const int*, const int*, const int*,
    const float* const*, const int*, const float* const*, const int*,
    float**, const int*, int, cudaStream_t, const float*);
