#include "gemm_tn_vbatch.cuh"
#include "gemm_nn_vbatch.cuh"
#include "dgemm_vbatch.h"
#include "source_base/module_device/device.h"

template<typename T>
void gemm_nn_vbatch(
    int max_m, int max_n, int max_k,
    int bucket_id,
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
    // bucket_id selected by caller based on per-item nw2 distribution.
    //
    // Two correctness traps (the old tune fell into both on case 008 / cal_gint_rho):
    //
    //   1. BLK_N must cover bxyz in a single tile. bxyz is typically 64 for production
    //      runs; using BLK_N=32 forces every NN block to be issued twice in N, doubling
    //      block count for the dominant compute path. BLK_N=64 it is.
    //
    //   2. Bucket 2 (nw2 > 16) catches ALL pairs whose nw2 is a TM-class atom, which
    //      includes Li-TM (M=7), O-TM (M=13), and TM-TM (M=27) -- M only ever depends on
    //      nw2, but K = nw1 is independent and varies. So the BLK_M tile must match the
    //      *bucket's* M=27, not the asymmetric pairs' actual M. A 32-wide BLK_M wastes
    //      ~16% of threads on TM-TM and was reg-limited on 3090; a 16-wide BLK_M splits
    //      M=27 as (16, 11) -> two blocks, *both 100% useful*.
    //
    // As with the TN path, this means buckets 1 and 2 want the same shape (BLK_M=16
    // covers nw2=13 with 81% useful and nw2=27 at 100% useful in 2 blocks). Bucket 0
    // keeps a distinct shape (BLK_M=8) because nw2 <= 8 doesn't justify a 16-wide tile.
    //
    // Per-bucket design (primary target: sm_86 / RTX 3090; cross-checked sm_80/sm_90):
    //   3090 limits per SM: 1536 threads, 16 blocks, ~100KB smem, ~4 FP64 FMA/cyc.
    //
    //   bucket 0 (nw2 <=  8): 128 thr, BLK 8x64x16, smem ~9.9KB.
    //                         3090: 9*128 = 1152 (75%); A100: 100%. BLK_N=64 covers
    //                         bxyz=64 in one tile (was 2 tiles); BLK_K=16 covers K=27
    //                         in 2 iters (TM-Li pairs).
    //   buckets 1 & 2 (nw2 > 8):
    //                         128 thr, BLK 16x64x16, smem ~10.9KB.
    //                         3090: 8*128 = 1024 (67%); A100: 100%. Single 16x64 tile
    //                         per block; M=27 splits as 2 blocks of (16, 11) at 100%
    //                         useful. K=27 is 2 K iters, K=13 is 1, K=7 is 1.
    //                         8 accum/thread keeps arith intensity high.
    if (bucket_id == 0) {
        //                         DIM_X,Y  BLK_M,N,K   DIM_XA,YA  DIM_XB,YB
        vbatched_gemm_nn_impl<T,   8, 16,   8, 64, 16,  8, 16,     8, 16>
            (max_m, max_n, m_d, n_d, k_d,
             A_array_d, lda_d, B_array_d, ldb_d,
             C_array_d, ldc_d, batchCount, stream, alpha);
    } else {
        vbatched_gemm_nn_impl<T,   8, 16,   16, 64, 16, 8, 16,     8, 16>
            (max_m, max_n, m_d, n_d, k_d,
             A_array_d, lda_d, B_array_d, ldb_d,
             C_array_d, ldc_d, batchCount, stream, alpha);
    }
}

template<typename T>
void gemm_tn_vbatch(
    int max_m, int max_n, int max_k,
    int bucket_id,
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
    // bucket_id selected by caller based on per-item max(nw1, nw2) distribution.
    //
    // CRITICAL: bucket 2 (max nw > 16) catches all pairs touching a TM-class atom,
    // including the asymmetric ones (TM-O = 13x27, TM-Li = 7x27, etc.). A 32x32 BLK
    // for bucket 2 wastes ~60% of threads on TM-O and ~80% on TM-Li because the
    // smaller dim only fills part of the tile. For mixed-nw systems like
    // Li27Ni9O54Mn9Co9 (case 008), TM-O is the single largest contributor to total
    // FMAs, so the wasted-thread cost dominates the K-iter savings of a wide tile.
    //
    // Conclusion: bucket 2 must use a 16x16 tile (matching the smaller dim of the
    // worst asymmetric pair). It then trivially also serves bucket 1, so both share
    // the same kernel shape -- bucket 0 is the only one with a distinct (smaller)
    // tile. The bucketing infrastructure stays in place because partition + per-
    // bucket grid sizing still helps avoid over-launching for bucket 0 problems.
    //
    // Per-bucket design (primary target: sm_86 / RTX 3090; cross-checked sm_80/sm_90):
    //   3090 limits per SM: 1536 threads, 16 blocks, ~100KB smem, ~4 FP64 FMA/cyc.
    //
    //   bucket 0 (max nw <=  8): 64 thr, BLK 8x8x32,    smem ~4.4KB.
    //                            3090: 16*64 = 1024 (67%); A100: 32*64 = 2048 (100%).
    //                            BLK_K=32 covers K=27 in a single tile (no main iter)
    //                            and halves K iters at K=64/80 vs BLK_K=16. Output
    //                            8x8 caps thread count at 64.
    //   buckets 1 & 2 (max nw > 8):
    //                            64 thr, BLK 16x16x32, smem ~8.6KB.
    //                            3090: 11*64 = 704  (46%); A100: 100%. 4 accum/thr
    //                            for high arith intensity; BLK_K=32 -> only 2 K iters
    //                            at K=64. This is the regression-note-validated tune
    //                            that achieved both 10.44s on Li/Ni/O/Mn/Co (case 008)
    //                            and 2.09s on Si216 (case 009-class) -- the joint best
    //                            measured. 16x16 tile handles asymmetric TM-O / TM-Li
    //                            pairs at full thread density (vs 32% / 18% with a
    //                            32x32 tile).
    if (bucket_id == 0) {
        //                         DIM_X,Y  BLK_M,N,K   DIM_XA,YA  DIM_XB,YB
        vbatched_gemm_tn_impl<T,   8, 8,    8, 8, 32,   8, 8,      8, 8>
            (max_m, max_n, m_d, n_d, k_d,
             A_array_d, lda_d, B_array_d, ldb_d,
             C_array_d, ldc_d, batchCount, stream, alpha);
    } else {
        vbatched_gemm_tn_impl<T,   8, 8,    16, 16, 32, 8, 8,      8, 8>
            (max_m, max_n, m_d, n_d, k_d,
             A_array_d, lda_d, B_array_d, ldb_d,
             C_array_d, ldc_d, batchCount, stream, alpha);
    }
}

// Explicit instantiations
template void gemm_nn_vbatch<double>(
    int, int, int, int, const int*, const int*, const int*,
    const double* const*, const int*, const double* const*, const int*,
    double**, const int*, int, cudaStream_t, const double*);

template void gemm_nn_vbatch<float>(
    int, int, int, int, const int*, const int*, const int*,
    const float* const*, const int*, const float* const*, const int*,
    float**, const int*, int, cudaStream_t, const float*);

template void gemm_tn_vbatch<double>(
    int, int, int, int, const int*, const int*, const int*,
    const double* const*, const int*, const double* const*, const int*,
    double**, const int*, int, cudaStream_t, const double*);

template void gemm_tn_vbatch<float>(
    int, int, int, int, const int*, const int*, const int*,
    const float* const*, const int*, const float* const*, const int*,
    float**, const int*, int, cudaStream_t, const float*);
