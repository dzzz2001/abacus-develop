#include "gemm_tn_vbatch.cuh"
#include "gemm_nn_vbatch.cuh"
#include "dgemm_vbatch.h"
#include "source_base/module_device/device.h"

// ----------------------------------------------------------------------------
// FP64-only big-tile dispatch (Phase V4)
// ----------------------------------------------------------------------------
// Pattern mirrors the C++11-compatible overload trick used in
// gint_vl.cpp / gint_rho.cpp (note in those files: "C++11-compatible
// alternative to if constexpr"). The non-template double overload is the
// preferred candidate when T = double; the template fallback returns
// false for every other dtype so the FP32 path stays untouched (per the
// V100 plan: 3090 FP32 was the test proxy and the big tile was not
// validated on FP32).
// Returns true if the big tile dispatched (caller should `return`).
// ----------------------------------------------------------------------------

inline bool nn_try_big_tile_(
    int m, int n, int k,
    const double* const* A_array_d, const int* lda_d,
    const double* const* B_array_d, const int* ldb_d,
    double** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream, const double* alpha)
{
    // 16x16 threads = 256, BLK_M=BLK_N=64, BLK_K=16. THR_M = THR_N = 4
    // -> 16 FMAs per inner step, 32 with VK=2. Loaders use DIM_*A=DIM_*B=16.
    if (n >= 48 && m >= 64) {
        vbatched_gemm_nn_impl<double,
                              /*DIM_X */ 16, /*DIM_Y */ 16,
                              /*BLK_M */ 64, /*BLK_N */ 64, /*BLK_K*/ 16,
                              /*DIM_XA*/ 16, /*DIM_YA*/ 16,
                              /*DIM_XB*/ 16, /*DIM_YB*/ 16>(
            m, n, k,
            A_array_d, lda_d, B_array_d, ldb_d,
            C_array_d, ldc_d, batchCount, stream, alpha);
        return true;
    }
    return false;
}

template <typename T>
inline bool nn_try_big_tile_(
    int /*m*/, int /*n*/, int /*k*/,
    const T* const* /*A*/, const int* /*lda*/,
    const T* const* /*B*/, const int* /*ldb*/,
    T** /*C*/, const int* /*ldc*/,
    int /*batch*/, cudaStream_t /*stream*/, const T* /*alpha*/)
{
    return false;
}

inline bool tn_try_big_tile_(
    int m, int n, int k,
    const double* const* A_array_d, const int* lda_d,
    const double* const* B_array_d, const int* ldb_d,
    double** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream, const double* alpha)
{
    // Axis flip vs NN: kernel M = wrapper n = nw2, kernel N = wrapper m = nw1.
    // Threshold is symmetric at 48 in both axes.
    // BLK_K=16 (not 32 as in the existing TN ladder) keeps the big-tile shmem
    // footprint at ~18 KB/block so 4 blocks/SM still fit on V100's 96 KB.
    if (n >= 48 && m >= 48) {
        vbatched_gemm_tn_impl<double,
                              /*DIM_X */ 16, /*DIM_Y */ 16,
                              /*BLK_M */ 64, /*BLK_N */ 64, /*BLK_K*/ 16,
                              /*DIM_XA*/ 16, /*DIM_YA*/ 16,
                              /*DIM_XB*/ 16, /*DIM_YB*/ 16>(
            m, n, k,
            A_array_d, lda_d, B_array_d, ldb_d,
            C_array_d, ldc_d, batchCount, stream, alpha);
        return true;
    }
    return false;
}

template <typename T>
inline bool tn_try_big_tile_(
    int /*m*/, int /*n*/, int /*k*/,
    const T* const* /*A*/, const int* /*lda*/,
    const T* const* /*B*/, const int* /*ldb*/,
    T** /*C*/, const int* /*ldc*/,
    int /*batch*/, cudaStream_t /*stream*/, const T* /*alpha*/)
{
    return false;
}

// ----------------------------------------------------------------------------
// Shape-exact dispatch
// ----------------------------------------------------------------------------
//
// The caller (phi_operator_gpu.cu) buckets atom pairs by (nw1, nw2) so every
// item in a batch has exactly the same (m, n, k). The scalars passed here are
// the *exact* per-matrix shapes (not upper bounds), which lets the tile
// ladder pick the tightest template and sizes the grid tightly (no
// over-launched blocks that short-circuit inside the kernel).
//
// Kernel-level dimension mapping (after the A/B swap inside
// vbatched_gemm_*_impl):
//
//   call    | wrapper m    | wrapper n   | wrapper k
//   --------|--------------|-------------|---------------
//   NN      | bxyz (large) | nw2 (small) | nw1 (small)
//   TN      | nw1 (small)  | nw2 (small) | bxyz (large)
//
// (m, n, k) flow through as scalars all the way down into the kernel, so
// there is no per-batchid M/N/K load and no fill-kernel scratch buffer.
// ----------------------------------------------------------------------------

template<typename T>
void gemm_nn_vbatch(
    int m, int n, int k,
    const T* const* A_array_d, const int* lda_d,
    const T* const* B_array_d, const int* ldb_d,
    T** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream,
    const T* alpha)
{
    // FP64 big tile (256-thread 64x64). Little's Law says V100 needs
    // ~300 in-flight FP64 FMAs/SM to saturate; the 16x16-thread 4x4
    // register tile puts 4096 FMAs/step/block in flight, so one block
    // already covers the pipe and the second one hides LDS latency.
    if (nn_try_big_tile_(m, n, k,
                         A_array_d, lda_d, B_array_d, ldb_d,
                         C_array_d, ldc_d, batchCount, stream, alpha))
    {
        return;
    }

    // 4 x 2 ladder (8 instantiations), tuned for V100 / A100:
    //   n (nw2 axis)  -> BLK_M in {8, 16, 32, 48}    (smallest full-cover)
    //   m (bxyz axis) -> BLK_N in {32, 64}           (larger-is-better)
    //   BLK_K fixed at 16                             (nw1 axis, <=27)
    //   DIM_X=8, DIM_Y=16 (128 threads/block, unchanged)
    //
    // Philosophy vs the prior tail-waste-min ladder:
    //
    // That ladder picked BLK_N by minimizing (tail_waste, grid_blocks)
    // lexicographically. It's the right objective on sm_86 consumer
    // Ampere (RTX 3090) where FP64 is 1/64 of FP32 -- every masked FMA
    // there is a full FP64-pipe-bound cycle, so minimizing launched
    // cells dominates.
    //
    // On V100 (sm_70) / A100 (sm_80) FP64 is a first-class pipe
    // (7.8 / 9.7 TFLOPS peak, ridge ~6-9 FLOP/B), and the inner loop
    // is LDS-bound for the nw1/nw2 ranges we see (ncu confirms L1/TEX
    // >= 95% on these tiles). The right objective flips to maximizing
    // per-block LDS reuse. The wide-LDS inner loop delivers
    //     FMA / LDS  =  VK * THR_M * THR_N / (THR_M + THR_N)
    // to the shmem pipe; for the scalar-K-tail regime (nw1 < BLK_K,
    // hit by nw1 <= 16 on NN) the VK factor drops out but the ratio
    // shape is the same. Rough V100 FP64 target ratio is 2 FMAs/LDS
    // (32 FP64-FMA/cycle/SM vs LDS.64 delivering one serving/cycle).
    //
    // At DIM=8x16, BLK_N=64 gives THR_N=4, THR_M=4 -> FMA/LDS = 2.0
    // (matched). BLK_N=32 drops it to 1.33 (LDS-bound, FP64 headroom
    // unused). So BLK_N=64 is strictly better for every bxyz >= 48;
    // only bxyz=27 still prefers BLK_N=32 to cap the N-axis mask waste
    // below 50%. The intermediate rungs {16, 48} are dropped: {32, 64}
    // covers bxyz in {27, 48, 64, 80, 100, 125} at its LDS-optimal
    // point in every case.
    //
    // BLK_M retains four rungs: the nw2 axis is tiny (<=44 in practice)
    // and a wrong-BLK_M costs twice -- masked FMAs *and* a wider sA
    // row load per K-step. The 48-rung is kept specifically for nw2=44
    // extended-basis atoms (Ti/Mn/Fe/Co/Ni/Cu/Zn/Zr/Ba); otherwise the
    // 32-rung falls off to a 2-tile grid at ~31% total waste.
    //
    // All eight (BLK_M, BLK_N) satisfy the kernel's BLK_M % DIM_X=0
    // and BLK_N % DIM_Y=0 constraints, and the tiny-tile {BLK_M=8,
    // BLK_N=32} rung still has THR_M=1 which compiles cleanly.
    #define NN_DISPATCH(BLK_M_, BLK_N_)                                    \
        vbatched_gemm_nn_impl<T, 8, 8, BLK_M_, BLK_N_, 16, 8, 8, 8, 8>( \
            m, n, k,                                                       \
            A_array_d, lda_d, B_array_d, ldb_d,                            \
            C_array_d, ldc_d, batchCount, stream, alpha)

    const int blk_m_tag = (n <= 8) ? 0 : (n <= 16) ? 1 : (n <= 32) ? 2 : 3;
    const int blk_n_tag = (m < 48) ? 0 : 1;  // {32, 64}

    switch (blk_m_tag * 2 + blk_n_tag) {
        case 0: NN_DISPATCH( 8, 32); break;
        case 1: NN_DISPATCH( 8, 64); break;
        case 2: NN_DISPATCH(16, 32); break;
        case 3: NN_DISPATCH(16, 64); break;
        case 4: NN_DISPATCH(32, 32); break;
        case 5: NN_DISPATCH(32, 64); break;
        case 6: NN_DISPATCH(48, 32); break;
        case 7: NN_DISPATCH(48, 64); break;
    }

    #undef NN_DISPATCH
}

template<typename T>
void gemm_tn_vbatch(
    int m, int n, int k,
    const T* const* A_array_d, const int* lda_d,
    const T* const* B_array_d, const int* ldb_d,
    T** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream,
    const T* alpha)
{
    // FP64 big tile (256-thread 64x64). Symmetric n>=48 && m>=48
    // because, after the kernel's A/B swap, both output axes are small
    // (kernel M = wrapper n = nw2, kernel N = wrapper m = nw1) and
    // neither is intrinsically larger than the other.
    if (tn_try_big_tile_(m, n, k,
                         A_array_d, lda_d, B_array_d, ldb_d,
                         C_array_d, ldc_d, batchCount, stream, alpha))
    {
        return;
    }

    // 4 x 4 ladder (16 instantiations), tuned for V100 / A100:
    //   n (nw2 axis) -> BLK_M in {8, 16, 32, 48}
    //   m (nw1 axis) -> BLK_N in {8, 16, 32, 48}
    //   BLK_K fixed at 32                        (bxyz axis)
    //   DIM_X=8, DIM_Y=8 (64 threads/block)
    //
    // Smallest-covering-tile selection, symmetric in both axes. This
    // is *not* the same choice as NN -- on TN both output axes are
    // small (nw1, nw2 in {4, 9, 13, 27, 44}) and neither is long
    // enough to amortize the "prefer bigger" BLK_N logic from NN.
    // Doubling BLK_* here would just push nw=4/9/13 cases off their
    // exact-fit tile into a 2-4x mask-waste regime with no LDS-reuse
    // upside (both axes of the output are already covered by one tile
    // in this regime; a bigger tile just adds masked FMAs).
    //
    // The 48-rung covers nw=44 extended-basis atoms (Ti/Mn/Fe/Co/Ni/
    // Cu/Zn/Zr/Ba) at ~8% mask waste per axis; without it those cases
    // fall to a 2-tile BLK=32 grid at ~52% cell-launch waste.
    //
    // BLK_K=32 (larger than NN's 16) because K = bxyz here is large
    // (27-125) and the K-axis tail wastes only shmem loads, never
    // masked FMAs on the output -- bxyz <= 32 fits in one K-tile,
    // larger bxyz wraps into 2-4 K-tiles. The modest __syncthreads()
    // overhead from more K-tiles is cheaper than doubling BLK_K and
    // forcing a re-tune of the `ra/rb` double-buffer register budget.
    //
    // All 16 (BLK_M, BLK_N) pairs are divisible by
    // DIM_X/DIM_Y/DIM_XA/DIM_YA/DIM_XB/DIM_YB = 8, so every
    // instantiation compiles to a valid kernel.
    #define TN_DISPATCH(BLK_M_, BLK_N_)                                 \
        vbatched_gemm_tn_impl<T, 4, 8, BLK_M_, BLK_N_, 32, 4, 8, 4, 8>( \
            m, n, k,                                                    \
            A_array_d, lda_d, B_array_d, ldb_d,                         \
            C_array_d, ldc_d, batchCount, stream, alpha)

    auto tag_for = [](int x) {
        return (x <= 8) ? 0 : (x <= 16) ? 1 : (x <= 32) ? 2 : 3;
    };
    const int blk_m_tag = tag_for(n); // kernel's M-dim grid -> wrapper n
    const int blk_n_tag = tag_for(m); // kernel's N-dim grid -> wrapper m

    switch (blk_m_tag * 4 + blk_n_tag) {
        case  0: TN_DISPATCH( 8,  8); break;
        case  1: TN_DISPATCH( 8, 16); break;
        case  2: TN_DISPATCH( 8, 32); break;
        case  3: TN_DISPATCH( 8, 48); break;
        case  4: TN_DISPATCH(16,  8); break;
        case  5: TN_DISPATCH(16, 16); break;
        case  6: TN_DISPATCH(16, 32); break;
        case  7: TN_DISPATCH(16, 48); break;
        case  8: TN_DISPATCH(32,  8); break;
        case  9: TN_DISPATCH(32, 16); break;
        case 10: TN_DISPATCH(32, 32); break;
        case 11: TN_DISPATCH(32, 48); break;
        case 12: TN_DISPATCH(48,  8); break;
        case 13: TN_DISPATCH(48, 16); break;
        case 14: TN_DISPATCH(48, 32); break;
        case 15: TN_DISPATCH(48, 48); break;
    }

    #undef TN_DISPATCH
}

// Explicit instantiations
template void gemm_nn_vbatch<double>(
    int, int, int,
    const double* const*, const int*, const double* const*, const int*,
    double**, const int*, int, cudaStream_t, const double*);

template void gemm_nn_vbatch<float>(
    int, int, int,
    const float* const*, const int*, const float* const*, const int*,
    float**, const int*, int, cudaStream_t, const float*);

template void gemm_tn_vbatch<double>(
    int, int, int,
    const double* const*, const int*, const double* const*, const int*,
    double**, const int*, int, cudaStream_t, const double*);

template void gemm_tn_vbatch<float>(
    int, int, int,
    const float* const*, const int*, const float* const*, const int*,
    float**, const int*, int, cudaStream_t, const float*);
