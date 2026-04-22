#include "gemm_tn_vbatch.cuh"
#include "gemm_nn_vbatch.cuh"
#include "dgemm_vbatch.h"
#include "source_base/module_device/device.h"

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
    // 3x4 ladder (12 instantiations), tuned for Ampere:
    //   n (nw2 axis)  -> BLK_M in {8, 16, 32}         (threshold ladder)
    //   m (bxyz axis) -> BLK_N in {16, 32, 48, 64}    (waste-minimizing)
    //   BLK_K fixed at 16                             (nw1 axis, <=13 here)
    //
    // After the A/B swap in vbatched_gemm_nn_impl, the kernel's N-axis covers
    // the bxyz dimension of the output C. Because M = bxyz is a runtime
    // scalar that varies across benchmark cases (27, 48, 64, 80, 100, 125)
    // and the register tile THR_N = BLK_N / DIM_Y is unrolled at compile
    // time, a BLK_N that does not evenly divide bxyz produces fully-computed
    // but mostly-masked tiles -- pure FMA waste on the under-full last
    // grid-y block.
    //
    // BLK_N is chosen by minimizing (tail_waste, grid_blocks)
    // lexicographically over the candidate set. This lands bxyz=48 on
    // BLK_N=48 (1 block, 0 waste) and bxyz=80/100 on BLK_N=16 (many blocks,
    // 0 waste), while bxyz=64/125 still pick BLK_N=64 and bxyz=27 still
    // picks BLK_N=32 (same 5-row tail as BLK_N=16 but 1 block instead of 2).
    // All four BLK_N values satisfy BLK_N % DIM_Y = BLK_N % DIM_YB = 0, so
    // the shmem-load loops and register tiles compile without changes.
    #define NN_DISPATCH(BLK_M_, BLK_N_)                                    \
        vbatched_gemm_nn_impl<T, 8, 16, BLK_M_, BLK_N_, 16, 8, 16, 8, 16>( \
            m, n, k,                                                       \
            A_array_d, lda_d, B_array_d, ldb_d,                            \
            C_array_d, ldc_d, batchCount, stream, alpha)

    // VERIFICATION PATCH 2026-04-22: extend BLK_M ladder to include 48 so
    // nw2 in (32, 48] (e.g. nw2=44 extended-basis atoms) lands on a 1-tile
    // grid with ~10% waste instead of a 2-tile BLK_M=32 grid with ~45% waste.
    const int blk_m_tag = (n <= 8) ? 0 : (n <= 16) ? 1 : (n <= 32) ? 2 : 3;

    int blk_n_tag = 0;
    {
        constexpr int cands[4] = {16, 32, 48, 64};
        int best_waste  = ((m + cands[0] - 1) / cands[0]) * cands[0] - m;
        int best_blocks = (m + cands[0] - 1) / cands[0];
        for (int i = 1; i < 4; ++i) {
            const int blocks = (m + cands[i] - 1) / cands[i];
            const int waste  = blocks * cands[i] - m;
            if (waste < best_waste ||
                (waste == best_waste && blocks < best_blocks)) {
                best_waste  = waste;
                best_blocks = blocks;
                blk_n_tag   = i;
            }
        }
    }

    switch (blk_m_tag * 4 + blk_n_tag) {
        case  0: NN_DISPATCH( 8, 16); break;
        case  1: NN_DISPATCH( 8, 32); break;
        case  2: NN_DISPATCH( 8, 48); break;
        case  3: NN_DISPATCH( 8, 64); break;
        case  4: NN_DISPATCH(16, 16); break;
        case  5: NN_DISPATCH(16, 32); break;
        case  6: NN_DISPATCH(16, 48); break;
        case  7: NN_DISPATCH(16, 64); break;
        case  8: NN_DISPATCH(32, 16); break;
        case  9: NN_DISPATCH(32, 32); break;
        case 10: NN_DISPATCH(32, 48); break;
        case 11: NN_DISPATCH(32, 64); break;
        case 12: NN_DISPATCH(48, 16); break;
        case 13: NN_DISPATCH(48, 32); break;
        case 14: NN_DISPATCH(48, 48); break;
        case 15: NN_DISPATCH(48, 64); break;
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
    // 3x3 ladder (9 instantiations), tuned for A100:
    //   n (nw2 axis) -> BLK_M in {8, 16, 32}
    //   m (nw1 axis) -> BLK_N in {8, 16, 32}
    //   BLK_K fixed at 32                        (bxyz axis)
    //
    // BLK_K is not split by bxyz: the K-axis tail wastes only shmem loads
    // (not FMAs), so a single BLK_K keeps the template table small while
    // still covering bxyz in [27, 125] via ceil(bxyz/32) K-tiles. bxyz=27
    // fits in one tile (5/32 = 16% load waste); larger bxyz wraps into
    // 2-4 K-tiles with modest __syncthreads() overhead.
    //
    // Block shape is DIM_X=8 x DIM_Y=8 (64 threads). Every (BLK_M, BLK_N)
    // pair is divisible by DIM_X/DIM_Y/DIM_XA/DIM_YA/DIM_XB/DIM_YB = 8,
    // so all nine combinations compile to valid kernels.
    #define TN_DISPATCH(BLK_M_, BLK_N_)                                 \
        vbatched_gemm_tn_impl<T, 8, 8, BLK_M_, BLK_N_, 32, 8, 8, 8, 8>( \
            m, n, k,                                                    \
            A_array_d, lda_d, B_array_d, ldb_d,                         \
            C_array_d, ldc_d, batchCount, stream, alpha)

    // VERIFICATION PATCH 2026-04-22: extend both BLK_M and BLK_N ladders up
    // to 48 so that nw in (32, 48] (extended-basis nw=44 atoms: Ti/Mn/Fe/Co/
    // Ni/Cu/Zn/Zr/Ba) lands on a 1-tile grid per axis (48^2 cells for 44^2
    // output, ~19% waste) instead of a 2-tile BLK_M=32 grid (64^2 cells,
    // ~52% waste).
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
