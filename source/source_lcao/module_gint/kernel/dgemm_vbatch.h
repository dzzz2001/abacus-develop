#pragma once

#include <cuda_runtime.h>

// Shape-exact batched GEMM dispatchers.
//
// Every (A_i, B_i, C_i) in the batch has exactly the same (m, n, k); the
// caller (phi_operator_gpu.cu) enforces this by bucketing atom pairs on
// (nw1, nw2). The scalars drive tile-ladder selection and grid sizing
// directly -- there is no "max" approximation left.
//
// `mnk_scratch_d` is a device-only scratch buffer of length >= 3*batchCount.
// The wrapper fills it with per-batchid M/N/K arrays (one fused fill kernel
// per call) before launching the underlying template kernel, which still
// indexes `M[batchid]` / `N[batchid]` / `K[batchid]` internally. Once
// gemm_{nn,tn}_vbatch.cuh is updated to take scalar M/N/K, the scratch
// parameter disappears and the fill launch with it.

// C(batch) = alpha * A(batch) * B(batch) + C(batch)
template<typename T>
void gemm_nn_vbatch(
    int m, int n, int k,
    int* mnk_scratch_d,
    const T* const* A_array_d, const int* lda_d,
    const T* const* B_array_d, const int* ldb_d,
    T** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream,
    const T* alpha = nullptr);

// C(batch) = alpha * A(batch)^T * B(batch) + C(batch)
template<typename T>
void gemm_tn_vbatch(
    int m, int n, int k,
    int* mnk_scratch_d,
    const T* const* A_array_d, const int* lda_d,
    const T* const* B_array_d, const int* ldb_d,
    T** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream,
    const T* alpha = nullptr);
