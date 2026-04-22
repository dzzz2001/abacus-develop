#pragma once

#include <cuda_runtime.h>

// Shape-exact batched GEMM dispatchers.
//
// Every (A_i, B_i, C_i) in the batch has exactly the same (m, n, k); the
// caller (phi_operator_gpu.cu) enforces this by bucketing atom pairs on
// (nw1, nw2). The scalars drive tile-ladder selection, grid sizing, and
// flow all the way through the kernel -- there is no per-batchid M/N/K
// indirection left.

// C(batch) = alpha * A(batch) * B(batch) + C(batch)
template<typename T>
void gemm_nn_vbatch(
    int m, int n, int k,
    const T* const* A_array_d, const int* lda_d,
    const T* const* B_array_d, const int* ldb_d,
    T** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream,
    const T* alpha = nullptr);

// C(batch) = alpha * A(batch)^T * B(batch) + C(batch)
template<typename T>
void gemm_tn_vbatch(
    int m, int n, int k,
    const T* const* A_array_d, const int* lda_d,
    const T* const* B_array_d, const int* ldb_d,
    T** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream,
    const T* alpha = nullptr);
