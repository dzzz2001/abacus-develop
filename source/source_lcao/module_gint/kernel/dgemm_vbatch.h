#pragma once

#include <cuda_runtime.h>

// Shape-based dispatch for batched GEMM kernels.
// The wrapper picks a kernel template instantiation from (max_m, max_n, max_k)
// of the sub-batch -- the caller does not need to know which template that maps
// to. Callers are still expected to feed shape-homogeneous sub-batches (e.g. by
// partitioning the full batch by orbital count) so that max_m / max_n / max_k
// stay close to the per-item values.

// Template version: C(batch_id) = alpha * A(batch_id) * B(batch_id) + C(batch_id)
template<typename T>
void gemm_nn_vbatch(
    int max_m, int max_n, int max_k,
    const int* m_d, const int* n_d, const int* k_d,
    const T* const* A_array_d, const int* lda_d,
    const T* const* B_array_d, const int* ldb_d,
    T** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream,
    const T* alpha = nullptr);

// Template version: C(batch_id) = alpha * A(batch_id)^T * B(batch_id) + C(batch_id)
template<typename T>
void gemm_tn_vbatch(
    int max_m, int max_n, int max_k,
    const int* m_d, const int* n_d, const int* k_d,
    const T* const* A_array_d, const int* lda_d,
    const T* const* B_array_d, const int* ldb_d,
    T** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream,
    const T* alpha = nullptr);
