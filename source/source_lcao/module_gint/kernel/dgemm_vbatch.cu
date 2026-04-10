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
    //
    // Dispatch key:
    //   max_n -> selects BLK_M (nw2 axis)
    //   max_m -> selects BLK_N (bxyz axis)
    //   max_k -> selects BLK_K (nw1 axis)
    //
    // The inner ladder (on max_m) steps BLK_N so each bxyz lands on a tile
    // that nearly fits it in one N-tile. The outer ladder (on max_n) picks
    // BLK_M: =8 when nw2 is all-small (Li/H/.. only), =16 otherwise.
    //
    // Tier 0 path (max_n <= 8) used to be a one-shot BLK 8x64x16. That
    // wastes 42% of the N axis at bxyz=27 and 25-56% on the boundary tile
    // at bxyz in {80, 100}. Mirroring the tier 1/2 bxyz ladder inside tier 0
    // restores a one-tile-per-matrix fit without changing the M/K tile.
    if (max_n <= 8) {
        //                             DIM_X,Y  BLK_M,N,K   DIM_XA,YA  DIM_XB,YB
        if (max_m <= 32) {
            vbatched_gemm_nn_impl<T,   8, 16,    8,  32, 16, 8, 16,    8, 16>
                (max_m, max_n, m_d, n_d, k_d,
                 A_array_d, lda_d, B_array_d, ldb_d,
                 C_array_d, ldc_d, batchCount, stream, alpha);
        } else if (max_m <= 48) {
            vbatched_gemm_nn_impl<T,   8, 16,    8,  48, 16, 8, 16,    8, 16>
                (max_m, max_n, m_d, n_d, k_d,
                 A_array_d, lda_d, B_array_d, ldb_d,
                 C_array_d, ldc_d, batchCount, stream, alpha);
        } else if (max_m <= 64) {
            vbatched_gemm_nn_impl<T,   8, 16,    8,  64, 16, 8, 16,    8, 16>
                (max_m, max_n, m_d, n_d, k_d,
                 A_array_d, lda_d, B_array_d, ldb_d,
                 C_array_d, ldc_d, batchCount, stream, alpha);
        } else if (max_m <= 80) {
            vbatched_gemm_nn_impl<T,   8, 16,    8,  80, 16, 8, 16,    8, 16>
                (max_m, max_n, m_d, n_d, k_d,
                 A_array_d, lda_d, B_array_d, ldb_d,
                 C_array_d, ldc_d, batchCount, stream, alpha);
        } else if (max_m <= 112) {
            vbatched_gemm_nn_impl<T,   8, 16,    8, 112, 16, 8, 16,    8, 16>
                (max_m, max_n, m_d, n_d, k_d,
                 A_array_d, lda_d, B_array_d, ldb_d,
                 C_array_d, ldc_d, batchCount, stream, alpha);
        } else {
            vbatched_gemm_nn_impl<T,   8, 16,    8, 128, 16, 8, 16,    8, 16>
                (max_m, max_n, m_d, n_d, k_d,
                 A_array_d, lda_d, B_array_d, ldb_d,
                 C_array_d, ldc_d, batchCount, stream, alpha);
        }
    } else if (max_m <= 32) {
        vbatched_gemm_nn_impl<T,   8, 16,   16,  32, 16, 8, 16,    8, 16>
            (max_m, max_n, m_d, n_d, k_d,
             A_array_d, lda_d, B_array_d, ldb_d,
             C_array_d, ldc_d, batchCount, stream, alpha);
    } else if (max_m <= 48) {
        vbatched_gemm_nn_impl<T,   8, 16,   16,  48, 16, 8, 16,    8, 16>
            (max_m, max_n, m_d, n_d, k_d,
             A_array_d, lda_d, B_array_d, ldb_d,
             C_array_d, ldc_d, batchCount, stream, alpha);
    } else if (max_m <= 64) {
        vbatched_gemm_nn_impl<T,   8, 16,   16,  64, 16, 8, 16,    8, 16>
            (max_m, max_n, m_d, n_d, k_d,
             A_array_d, lda_d, B_array_d, ldb_d,
             C_array_d, ldc_d, batchCount, stream, alpha);
    } else if (max_m <= 80) {
        vbatched_gemm_nn_impl<T,   8, 16,   16,  80, 16, 8, 16,    8, 16>
            (max_m, max_n, m_d, n_d, k_d,
             A_array_d, lda_d, B_array_d, ldb_d,
             C_array_d, ldc_d, batchCount, stream, alpha);
    } else if (max_m <= 112) {
        vbatched_gemm_nn_impl<T,   8, 16,   16, 112, 16, 8, 16,    8, 16>
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
    //   max_n -> selects BLK_M (nw2 axis)
    //   max_m -> selects BLK_N (nw1 axis)
    //   max_k -> selects BLK_K (bxyz axis)
    const int max_mn = max_m > max_n ? max_m : max_n;
    if (max_mn <= 8) {
        //                         DIM_X,Y  BLK_M,N,K   DIM_XA,YA  DIM_XB,YB
        vbatched_gemm_tn_impl<T,   8, 8,     8,  8, 32, 8, 8,      8, 8>
            (max_m, max_n, m_d, n_d, k_d,
             A_array_d, lda_d, B_array_d, ldb_d,
             C_array_d, ldc_d, batchCount, stream, alpha);
    } else if (max_m <= 8) {
        // Tier Am: nw1 small, nw2 large. BLK_N=8 on the nw1 axis.
        if (max_k <= 64) {
            vbatched_gemm_tn_impl<T,   8, 8,    16,  8, 32, 8, 8,      8, 8>
                (max_m, max_n, m_d, n_d, k_d,
                 A_array_d, lda_d, B_array_d, ldb_d,
                 C_array_d, ldc_d, batchCount, stream, alpha);
        } else {
            vbatched_gemm_tn_impl<T,   8, 8,    16,  8, 64, 8, 8,      8, 8>
                (max_m, max_n, m_d, n_d, k_d,
                 A_array_d, lda_d, B_array_d, ldb_d,
                 C_array_d, ldc_d, batchCount, stream, alpha);
        }
    } else if (max_n <= 8) {
        // Tier An: nw2 small, nw1 large. BLK_M=8 on the nw2 axis.
        if (max_k <= 64) {
            vbatched_gemm_tn_impl<T,   8, 8,     8, 16, 32, 8, 8,      8, 8>
                (max_m, max_n, m_d, n_d, k_d,
                 A_array_d, lda_d, B_array_d, ldb_d,
                 C_array_d, ldc_d, batchCount, stream, alpha);
        } else {
            vbatched_gemm_tn_impl<T,   8, 8,     8, 16, 64, 8, 8,      8, 8>
                (max_m, max_n, m_d, n_d, k_d,
                 A_array_d, lda_d, B_array_d, ldb_d,
                 C_array_d, ldc_d, batchCount, stream, alpha);
        }
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
