#include "gemm_tn_vbatch.cuh"
#include "gemm_nn_vbatch.cuh"
#include "dgemm_vbatch.h"
#include "source_base/module_device/device.h"

template<typename T>
void gemm_nn_vbatch(
    int max_m, int max_n, int max_k,
    int avg_m, int avg_n,
    const int* m_d, const int* n_d, const int* k_d,
    const T* const* A_array_d, const int* lda_d,
    const T* const* B_array_d, const int* ldb_d,
    T** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream,
    const T* alpha)
{
    // NN dimension mapping (after A/B swap in _impl):
    //   kernel M = n (nw, small 4-27), kernel N = m (bxyz, large 27-80), kernel K = k (nw, small 4-27)
    // Optimal: small BLK_M (nw), large BLK_N (bxyz), small BLK_K (nw).
    // Dispatch on avg_n which reflects the nw (kernel M) dimension.
    if (avg_n <= 8) {
        //                         DIM_X,Y  BLK_M,N,K  DIM_XA,YA  DIM_XB,YB
        vbatched_gemm_nn_impl<T,   8, 4,    8, 32, 8,  8, 4,      8, 4>
            (max_m, max_n, m_d, n_d, k_d,
             A_array_d, lda_d, B_array_d, ldb_d,
             C_array_d, ldc_d, batchCount, stream, alpha);
    } else if (avg_n <= 16) {
        vbatched_gemm_nn_impl<T,   8, 8,    16, 32, 8, 8, 8,      8, 8>
            (max_m, max_n, m_d, n_d, k_d,
             A_array_d, lda_d, B_array_d, ldb_d,
             C_array_d, ldc_d, batchCount, stream, alpha);
    } else {
        vbatched_gemm_nn_impl<T,   16, 8,   32, 64, 16, 16, 8,    16, 8>
            (max_m, max_n, m_d, n_d, k_d,
             A_array_d, lda_d, B_array_d, ldb_d,
             C_array_d, ldc_d, batchCount, stream, alpha);
    }
}

template<typename T>
void gemm_tn_vbatch(
    int max_m, int max_n, int max_k,
    int avg_m, int avg_n,
    const int* m_d, const int* n_d, const int* k_d,
    const T* const* A_array_d, const int* lda_d,
    const T* const* B_array_d, const int* ldb_d,
    T** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream,
    const T* alpha)
{
    // TN dimension mapping (after A/B swap in _impl):
    //   kernel M = n (nw, small 4-27), kernel N = m (nw, small 4-27), kernel K = k (bxyz, large 27-80)
    // Optimal: small BLK_M (nw), small BLK_N (nw), large BLK_K (bxyz).
    // Dispatch on max(avg_m, avg_n) since both are nw values.
    const int avg_mn = (avg_m > avg_n) ? avg_m : avg_n;

    if (avg_mn <= 8) {
        //                         DIM_X,Y  BLK_M,N,K   DIM_XA,YA  DIM_XB,YB
        vbatched_gemm_tn_impl<T,   8, 4,    8, 8, 32,   8, 4,      8, 4>
            (max_m, max_n, m_d, n_d, k_d,
             A_array_d, lda_d, B_array_d, ldb_d,
             C_array_d, ldc_d, batchCount, stream, alpha);
    } else if (avg_mn <= 16) {
        // 64 threads to keep register pressure low (~48 regs) with BLK_K=32
        vbatched_gemm_tn_impl<T,   8, 8,    16, 16, 32, 8, 8,      8, 8>
            (max_m, max_n, m_d, n_d, k_d,
             A_array_d, lda_d, B_array_d, ldb_d,
             C_array_d, ldc_d, batchCount, stream, alpha);
    } else {
        vbatched_gemm_tn_impl<T,   16, 8,   32, 32, 32, 16, 8,     16, 8>
            (max_m, max_n, m_d, n_d, k_d,
             A_array_d, lda_d, B_array_d, ldb_d,
             C_array_d, ldc_d, batchCount, stream, alpha);
    }
}

// Explicit instantiations
template void gemm_nn_vbatch<double>(
    int, int, int, int, int, const int*, const int*, const int*,
    const double* const*, const int*, const double* const*, const int*,
    double**, const int*, int, cudaStream_t, const double*);

template void gemm_nn_vbatch<float>(
    int, int, int, int, int, const int*, const int*, const int*,
    const float* const*, const int*, const float* const*, const int*,
    float**, const int*, int, cudaStream_t, const float*);

template void gemm_tn_vbatch<double>(
    int, int, int, int, int, const int*, const int*, const int*,
    const double* const*, const int*, const double* const*, const int*,
    double**, const int*, int, cudaStream_t, const double*);

template void gemm_tn_vbatch<float>(
    int, int, int, int, int, const int*, const int*, const int*,
    const float* const*, const int*, const float* const*, const int*,
    float**, const int*, int, cudaStream_t, const float*);
