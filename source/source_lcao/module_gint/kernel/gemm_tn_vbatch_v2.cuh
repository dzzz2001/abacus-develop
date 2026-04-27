#ifndef GEMM_TN_VBATCH_V2_CUH
#define GEMM_TN_VBATCH_V2_CUH
#include <cuda_runtime.h>

#include "gemm_mma_helpers.cuh"
#include "gemm_tn_vbatch_scalar.cuh"   // for gemm_tn_vbatch_scalar_dispatch fallback

// ----------------------------------------------------------------------------
// gemm_tn_vbatch_v2 — A100-targeted row-major batched FP64 GEMM (TN flavor).
// Computes C = α A^T B + C with all matrices row-major.
//
// Phase 1: stub. Forwards to the scalar dispatch. See gemm_nn_vbatch_v2.cuh
// for the redesign overview.
//
// Phase 3 (planned): native row-major kernel:
//   - One CTA per matrix. Both A (K x M) and B (K x N) shmem-resident.
//     For K = 125, M = N = 50 the combined shmem is ~100 KB and needs a
//     one-time cudaFuncSetAttribute opt-in to MaxDynamicSharedMemorySize.
//   - Inner loop mma.f64 on sm_80/90; scalar FMA fallback for FP32.
//   - Single-stage cp.async in Phase 3; promoted to two-stage ping-pong for
//     K ≥ 64 in Phase 4.
//
// External contract (row-major, native — no swap):
//   A is (K x M) at A[i*lda + j]; lda = phi_len_mgrid.
//   B is (K x N) at B[i*ldb + j]; ldb = phi_len_mgrid.
//   C is (M x N) at C[i*ldc + j]; ldc = nw2. Atomic accumulate.
// ----------------------------------------------------------------------------

template<typename T>
void gemm_tn_vbatch_v2_dispatch(
    int m, int n, int k,
    const T* const* A_array_d, const int* lda_d,
    const T* const* B_array_d, const int* ldb_d,
    T** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream,
    const T* alpha)
{
    // Phase 1 stub: forward to scalar dispatch.
    gemm_tn_vbatch_scalar_dispatch<T>(
        m, n, k,
        A_array_d, lda_d, B_array_d, ldb_d,
        C_array_d, ldc_d, batchCount, stream, alpha);
}

#endif // GEMM_TN_VBATCH_V2_CUH
