#ifndef GEMM_NN_VBATCH_V2_CUH
#define GEMM_NN_VBATCH_V2_CUH
#include <cuda_runtime.h>
#include <type_traits>

#include "gemm_mma_helpers.cuh"
#include "gemm_nn_vbatch_scalar.cuh"   // FP32 / sm_70/75 fallback

// ============================================================================
// gemm_nn_vbatch_v2 — A100-targeted row-major batched FP64 GEMM (Phase 2).
//
// External contract (row-major, no swap):
//   A is (M x K) at A[i*lda + j]; lda = phi_len_mgrid (caller-provided).
//   B is (K x N) at B[i*ldb + j]; ldb = nw2.
//   C is (M x N) at C[i*ldc + j]; ldc = phi_len_mgrid. Atomic accumulate
//   with per-batch alpha (preserves phi_mul_dm's is_symm semantics —
//   phi_operator_gpu.cu lines 425-428).
//
// Kernel structure:
//   - One CTA per matrix; threads = 128 = 4 warps.
//   - Each warp owns 16 m-rows of the output tile (BLK_M = 64 fixed).
//     Within a warp the 16 rows are split into two m=8 stripes for the
//     m8n8k4 mma shape.
//   - Tile rungs selected at host dispatch by N:
//       N <= 16 : (BLK_M, BLK_N) = (64, 16)   covers nw2 in {4, 9, 13, 16}
//       N >  16 : (BLK_M, BLK_N) = (64, 56)   covers nw2 in {25, 27, 44, 50}
//   - Inner loop: mma.sync.aligned.m8n8k4.row.col.f64 on sm_80+; per-lane
//     scalar `ld.shared.f64` fragment loads (no ldmatrix until Phase 4 perf).
//   - For M > 64 (bxyz = 100, 125), the CTA loops over m-strips internally;
//     sB stays resident in shmem across strips so B is only read once.
//   - K-tail: shmem K-pad rounded up to next multiple of 4; tail rows zero-
//     filled at load time so mma reads zeros for k >= K (bit-exact w.r.t.
//     truncating the K-sum).
//
// FP32 + sm_70/75 FP64: forwarded to gemm_nn_vbatch_scalar_dispatch<T>.
//
// Lane fragment layout (PTX, m8n8k4.row.col.f64; PTX ISA 8.5 Table 38):
//   For lane (lr = lane >> 2 ∈ 0..7,  lc = lane & 3 ∈ 0..3):
//     A frag (8 m × 4 k row-major): rA = A[lr, lc]            (1 double / lane)
//     B frag (4 k × 8 n col-major): rB = B[k=lc, n=lr]        (1 double / lane)
//     D frag (8 m × 8 n row-major): rD[0] = D[lr, 2*lc],
//                                   rD[1] = D[lr, 2*lc + 1]   (2 doubles / lane)
//
//   Per warp the 16 m-rows are covered by two stripes (stripe s ∈ {0, 1}
//   maps to global rows m_base + warp_row + s*8 + lr). The B frag is
//   independent of stripe, so we issue 2 mma calls per (k_step, n_frag),
//   one per stripe, sharing rB across them.
// ============================================================================

namespace gemm_nn_v2 {

// ----------------------------------------------------------------------------
// FP64 mma kernel.
// Shmem layout (no PAD; bank-conflict avoidance is a Phase 4 perf concern):
//   sA [BLK_M  x K_pad]  M-major, K-inner   sA[i * K_pad + k]
//   sB [K_pad  x BLK_N]  K-major, N-inner   sB[k * BLK_N + n]
//   K_pad = round_up(K, 4). Tail rows/cols zero-filled at load time.
// ----------------------------------------------------------------------------
template <int BLK_M, int BLK_N>
__launch_bounds__(128, 1)
__global__ void mma_fp64_kernel(
    int M, int N, int K,
    const double* const* __restrict__ A_array,
    const int* __restrict__ lda_array,
    const double* const* __restrict__ B_array,
    const int* __restrict__ ldb_array,
    double** __restrict__ C_array,
    const int* __restrict__ ldc_array,
    const double* __restrict__ alpha_array)
{
    static_assert(BLK_M == 64,
                  "BLK_M must be 64 (4 warps * 16 m-rows / warp = 2 m=8 stripes / warp)");
    static_assert(BLK_N % 8 == 0,
                  "BLK_N must be a multiple of 8 (mma n-frag = 8)");

    constexpr int N_FRAGS = BLK_N / 8;
    constexpr int M_STRIPES = 2;   // two m=8 stripes per warp's 16-row band

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int warp_row = warp_id * 16;   // 0, 16, 32, 48
    const int lr      = lane >> 2;        // 0..7
    const int lc      = lane & 3;         // 0..3

    const int batch = blockIdx.x;
    const double* __restrict__ A = A_array[batch];
    const double* __restrict__ B = B_array[batch];
    double* __restrict__ C       = C_array[batch];
    const int LDA = lda_array[batch];
    const int LDB = ldb_array[batch];
    const int LDC = ldc_array[batch];
    const double alpha = (alpha_array != nullptr) ? alpha_array[batch] : 1.0;

    const int K_pad   = (K + 3) & ~3;
    const int K_steps = K_pad >> 2;
    const int n_strips = (M + BLK_M - 1) / BLK_M;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    double* sA = reinterpret_cast<double*>(smem_raw);
    double* sB = sA + BLK_M * K_pad;

    // -------- Stage 0: load sB once. Persistent across all m-strips. ---------
    // sB[k * BLK_N + n] = (k < K && n < N) ? B[k*LDB + n] : 0.0
    {
        const int total = K_pad * BLK_N;
        #pragma unroll 1
        for (int idx = tid; idx < total; idx += 128)
        {
            int k = idx / BLK_N;
            int n = idx - k * BLK_N;
            double v = 0.0;
            if (k < K && n < N) v = B[k * LDB + n];
            sB[k * BLK_N + n] = v;
        }
    }

    for (int s = 0; s < n_strips; ++s)
    {
        const int m_base = s * BLK_M;

        // -------- Stage 1: load sA strip. ------------------------------------
        // sA[i * K_pad + k] = (m_g < M && k < K) ? A[m_g*LDA + k] : 0.0
        // m_g = m_base + i.
        {
            const int total = BLK_M * K_pad;
            #pragma unroll 1
            for (int idx = tid; idx < total; idx += 128)
            {
                int i = idx / K_pad;
                int k = idx - i * K_pad;
                double v = 0.0;
                int m_g = m_base + i;
                if (m_g < M && k < K) v = A[m_g * LDA + k];
                sA[i * K_pad + k] = v;
            }
        }

        __syncthreads();

        // -------- Stage 2: mma inner loop. -----------------------------------
        // Per-lane register footprint:
        //   rA[2]                    : A frag for both m-stripes, 1 double each
        //   rB                       : B frag (shared across stripes)
        //   rC[M_STRIPES][N_FRAGS][2]: C frag, lives across all k_steps
        double rC[M_STRIPES][N_FRAGS][2];
        #pragma unroll
        for (int ms = 0; ms < M_STRIPES; ++ms)
        {
            #pragma unroll
            for (int nf = 0; nf < N_FRAGS; ++nf)
            {
                rC[ms][nf][0] = 0.0;
                rC[ms][nf][1] = 0.0;
            }
        }

        for (int k_step = 0; k_step < K_steps; ++k_step)
        {
            const int k_base = k_step * 4;

            // A frag: lane(lr, lc) reads A[m=stripe*8+lr, k=k_base+lc].
            // sA[i * K_pad + k] => index = (warp_row + ms*8 + lr)*K_pad + (k_base+lc)
            double rA[M_STRIPES];
            rA[0] = sA[(warp_row + lr)     * K_pad + (k_base + lc)];
            rA[1] = sA[(warp_row + lr + 8) * K_pad + (k_base + lc)];

            #pragma unroll
            for (int nf = 0; nf < N_FRAGS; ++nf)
            {
                // B frag: lane(lr, lc) reads B[k=k_base+lc, n=nf*8+lr].
                // sB[k * BLK_N + n] => index = (k_base+lc)*BLK_N + (nf*8+lr).
                const double rB = sB[(k_base + lc) * BLK_N + (nf * 8 + lr)];

                mma_m8n8k4_f64(rC[0][nf], rA[0], rB);
                mma_m8n8k4_f64(rC[1][nf], rA[1], rB);
            }
        }

        // -------- Stage 3: atomicAdd rC into C, masking m and n. -------------
        // Per-lane writes 4 values per (stripe, n_frag) — but a stripe's
        // contribution is 2 (lane (lr, lc) -> D[lr, 2*lc] and D[lr, 2*lc+1]).
        // Across 2 stripes that's 4 C values per (lane, n_frag).
        const int n_left_lc  = 2 * lc;
        const int n_right_lc = n_left_lc + 1;

        #pragma unroll
        for (int ms = 0; ms < M_STRIPES; ++ms)
        {
            const int m_g = m_base + warp_row + ms * 8 + lr;
            if (m_g >= M) continue;

            #pragma unroll
            for (int nf = 0; nf < N_FRAGS; ++nf)
            {
                const int n_left_g  = nf * 8 + n_left_lc;
                const int n_right_g = nf * 8 + n_right_lc;

                if (n_left_g  < N)
                    atomicAdd(C + m_g * LDC + n_left_g,  rC[ms][nf][0] * alpha);
                if (n_right_g < N)
                    atomicAdd(C + m_g * LDC + n_right_g, rC[ms][nf][1] * alpha);
            }
        }

        // sB persists across strips; sA will be overwritten by the next strip.
        if (s + 1 < n_strips) __syncthreads();
    }
}

// ----------------------------------------------------------------------------
// Host launcher for the FP64 mma kernel.
//
// Shmem opt-in: at the (64, 56) rung with K=50 (nw=50) the dynamic shmem hits
// ~49 KB, which exceeds the default 48 KB per-block cap. cudaFuncSetAttribute
// raises the cap to 64 KB — covers our worst NN case (~50 KB) with margin and
// is well within every sm_80+ opt-in limit (sm_80: 163 KB, sm_86: 99 KB,
// sm_90: 227 KB). 100 KB would actually fail on sm_86, where the per-block
// cap is 99 KB; 64 KB is the safe portable value. Call is idempotent and
// gated by a per-instantiation `static int` flag so it runs once per symbol.
// ----------------------------------------------------------------------------
template <int BLK_M, int BLK_N>
inline void launch_mma_fp64(
    int m, int n, int k,
    const double* const* A_array_d, const int* lda_d,
    const double* const* B_array_d, const int* ldb_d,
    double** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream, const double* alpha)
{
    if (batchCount == 0) return;
    const int K_pad = (k + 3) & ~3;
    const size_t shmem_bytes =
        (static_cast<size_t>(BLK_M) * K_pad + static_cast<size_t>(K_pad) * BLK_N) * sizeof(double);

    static int shmem_opted_in = 0;
    if (!shmem_opted_in)
    {
        cudaError_t e = cudaFuncSetAttribute(
            reinterpret_cast<const void*>(&mma_fp64_kernel<BLK_M, BLK_N>),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            64 * 1024);
        if (e != cudaSuccess)
        {
            fprintf(stderr,
                    "gemm_nn_v2: cudaFuncSetAttribute(MaxDynamicSharedMemorySize) "
                    "failed for BLK_M=%d BLK_N=%d: %s\n",
                    BLK_M, BLK_N, cudaGetErrorString(e));
        }
        shmem_opted_in = 1;
    }

    dim3 grid(batchCount);
    dim3 block(128);

    mma_fp64_kernel<BLK_M, BLK_N>
        <<<grid, block, shmem_bytes, stream>>>(
            m, n, k,
            A_array_d, lda_d, B_array_d, ldb_d,
            C_array_d, ldc_d, alpha);
    CHECK_LAST_CUDA_ERROR("gemm_nn_v2::mma_fp64_kernel launch");
}

// ----------------------------------------------------------------------------
// FP64-only v2 attempt — selects the rung by N. Returns true on dispatch.
// Mirrors the C++14 SFINAE pattern in nn_try_big_tile_scalar_ (gemm_nn_vbatch_
// scalar.cuh): non-template double overload is the preferred candidate when
// T = double; the template fallback returns false for any other dtype so the
// FP32 path falls through to the existing scalar dispatcher (kept untouched
// in Phase 2 — the existing FP32 kernel is already well-tuned and HBM-bound).
// ----------------------------------------------------------------------------
inline bool nn_try_v2_(
    int m, int n, int k,
    const double* const* A_array_d, const int* lda_d,
    const double* const* B_array_d, const int* ldb_d,
    double** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream, const double* alpha)
{
    if (n <= 16)
    {
        launch_mma_fp64<64, 16>(
            m, n, k,
            A_array_d, lda_d, B_array_d, ldb_d,
            C_array_d, ldc_d, batchCount, stream, alpha);
    }
    else
    {
        launch_mma_fp64<64, 56>(
            m, n, k,
            A_array_d, lda_d, B_array_d, ldb_d,
            C_array_d, ldc_d, batchCount, stream, alpha);
    }
    return true;
}

template <typename T>
inline bool nn_try_v2_(
    int /*m*/, int /*n*/, int /*k*/,
    const T* const* /*A*/, const int* /*lda*/,
    const T* const* /*B*/, const int* /*ldb*/,
    T** /*C*/, const int* /*ldc*/,
    int /*batch*/, cudaStream_t /*stream*/, const T* /*alpha*/)
{
    return false;
}

}  // namespace gemm_nn_v2

template<typename T>
void gemm_nn_vbatch_v2_dispatch(
    int m, int n, int k,
    const T* const* A_array_d, const int* lda_d,
    const T* const* B_array_d, const int* ldb_d,
    T** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream,
    const T* alpha)
{
    if (gemm_nn_v2::nn_try_v2_(m, n, k,
                                A_array_d, lda_d, B_array_d, ldb_d,
                                C_array_d, ldc_d, batchCount, stream, alpha))
    {
        return;
    }
    // Fallback: FP32 (Phase 2 keeps the existing scalar tile-ladder; the v2
    // FP32 path is deferred — its scalar FMA inner loop is structurally
    // similar to the existing kernel's, and HBM-boundness means the existing
    // tuning already extracts most of the ceiling) and any non-FP64 dtype
    // routes through the scalar dispatcher.
    gemm_nn_vbatch_scalar_dispatch<T>(
        m, n, k,
        A_array_d, lda_d, B_array_d, ldb_d,
        C_array_d, ldc_d, batchCount, stream, alpha);
}

#endif // GEMM_NN_VBATCH_V2_CUH
