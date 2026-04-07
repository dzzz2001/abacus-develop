#pragma once

#include <cuda_runtime.h>

// Template version: C(batch_id) = alpha * A(batch_id) * B(batch_id) + C(batch_id)
// avg_m/avg_n: average M/N across batch items, used to select tile size.
// max_m/max_n: still used for grid sizing (must cover the largest item).
template<typename T>
void gemm_nn_vbatch(
    int max_m, int max_n, int max_k,
    int avg_m, int avg_n,
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
    int avg_m, int avg_n,
    const int* m_d, const int* n_d, const int* k_d,
    const T* const* A_array_d, const int* lda_d,
    const T* const* B_array_d, const int* ldb_d,
    T** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream,
    const T* alpha = nullptr);

// Legacy double-only aliases for backward compatibility
inline void dgemm_nn_vbatch(
    int max_m, int max_n, int max_k,
    int avg_m, int avg_n,
    const int* m_d, const int* n_d, const int* k_d,
    const double* const* A_array_d, const int* lda_d,
    const double* const* B_array_d, const int* ldb_d,
    double** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream,
    const double* alpha = nullptr)
{
    gemm_nn_vbatch<double>(max_m, max_n, max_k, avg_m, avg_n,
        m_d, n_d, k_d, A_array_d, lda_d, B_array_d, ldb_d,
        C_array_d, ldc_d, batchCount, stream, alpha);
}

inline void dgemm_tn_vbatch(
    int max_m, int max_n, int max_k,
    int avg_m, int avg_n,
    const int* m_d, const int* n_d, const int* k_d,
    const double* const* A_array_d, const int* lda_d,
    const double* const* B_array_d, const int* ldb_d,
    double** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream,
    const double* alpha = nullptr)
{
    gemm_tn_vbatch<double>(max_m, max_n, max_k, avg_m, avg_n,
        m_d, n_d, k_d, A_array_d, lda_d, B_array_d, ldb_d,
        C_array_d, ldc_d, batchCount, stream, alpha);
}