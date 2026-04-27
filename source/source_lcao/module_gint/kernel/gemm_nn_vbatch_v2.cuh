#ifndef GEMM_NN_VBATCH_V2_CUH
#define GEMM_NN_VBATCH_V2_CUH
#include <cuda_runtime.h>

#include "gemm_mma_helpers.cuh"
#include "gemm_nn_vbatch_scalar.cuh"   // for gemm_nn_vbatch_scalar_dispatch fallback

// ----------------------------------------------------------------------------
// gemm_nn_vbatch_v2 — A100-targeted row-major batched FP64 GEMM.
//
// Phase 1: stub. The dispatcher forwards directly to the scalar dispatch
// (`gemm_nn_vbatch_scalar_dispatch<T>`). No behavior change vs the existing
// kernel — the routing is wired so Phase 2 can swap the body in without
// touching dgemm_vbatch.cu.
//
// Phase 2 (planned): native row-major kernel:
//   - One CTA per matrix. Output tile (BLK_M, BLK_N) covers the whole matrix.
//   - B (K x N, ≤ 50x50) shmem-resident; A (M x K, ≤ 125x50) tiled in M.
//   - Inner loop: mma.sync.aligned.m16n8k8.row.col.f64 on sm_80/90;
//     scalar FMA on FP32 and the unreached non-TC FP64 fallback path.
//   - cp.async load (single-stage; A and B both prefetched at CTA entry).
//   - atomicAdd to row-major C with per-batch alpha (preserves the
//     phi_mul_dm is_symm semantics in phi_operator_gpu.cu lines 425-428).
//
// External contract (row-major, native — no swap):
//   A is (M x K) at A[i*lda + j]; lda = phi_len_mgrid (from caller).
//   B is (K x N) at B[i*ldb + j]; ldb = nw2.
//   C is (M x N) at C[i*ldc + j]; ldc = phi_len_mgrid. Atomic accumulate.
// ----------------------------------------------------------------------------

template<typename T>
void gemm_nn_vbatch_v2_dispatch(
    int m, int n, int k,
    const T* const* A_array_d, const int* lda_d,
    const T* const* B_array_d, const int* ldb_d,
    T** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream,
    const T* alpha)
{
    // Phase 1 stub: forward to scalar dispatch. Behavior is identical to
    // the pre-redesign code path. Phase 2 will replace this body with the
    // shmem-resident TC kernel launch.
    gemm_nn_vbatch_scalar_dispatch<T>(
        m, n, k,
        A_array_d, lda_d, B_array_d, ldb_d,
        C_array_d, ldc_d, batchCount, stream, alpha);
}

#endif // GEMM_NN_VBATCH_V2_CUH
