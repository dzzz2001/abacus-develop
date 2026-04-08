#pragma once

#include <cuda_runtime.h>

// Bucket-based dispatch for batched GEMM kernels.
// The caller partitions its batch by problem size and issues one launch per
// non-empty bucket. bucket_id selects the kernel template instantiation:
//   0 = small  (key <= 8)
//   1 = medium (8 < key <= 16)
//   2 = large  (key > 16)
// where "key" is nw2 for the NN path and max(nw1, nw2) for the TN path.
// max_m / max_n are still used for grid sizing and must cover the largest
// item in this sub-batch.

// Template version: C(batch_id) = alpha * A(batch_id) * B(batch_id) + C(batch_id)
template<typename T>
void gemm_nn_vbatch(
    int max_m, int max_n, int max_k,
    int bucket_id,
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
    int bucket_id,
    const int* m_d, const int* n_d, const int* k_d,
    const T* const* A_array_d, const int* lda_d,
    const T* const* B_array_d, const int* ldb_d,
    T** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream,
    const T* alpha = nullptr);
