#ifndef GEMM_TN_VBATCH_V2_CUH
#define GEMM_TN_VBATCH_V2_CUH
#include <cuda_runtime.h>
#include <type_traits>
#include <cstdio>

#include "gemm_mma_helpers.cuh"
#include "gemm_tn_vbatch_scalar.cuh"   // FP32 / sm_70/75 fallback

// ============================================================================
// gemm_tn_vbatch_v2 — A100-targeted row-major batched FP64 GEMM (Phase 3).
//
// Computes C = α A^T B + C with row-major operands and atomic accumulate.
//
// External contract (row-major, no swap):
//   A is (K x M) at A[i*lda + j]; lda = phi_len_mgrid (caller-provided).
//   B is (K x N) at B[i*ldb + j]; ldb = phi_len_mgrid.
//   C is (M x N) at C[i*ldc + j]; ldc = nw2. Atomic accumulate with per-batch
//   alpha (preserves phi_mul_phi's cross-call accumulation semantics —
//   phi_operator_gpu.cu line 449).
//
// Kernel structure:
//   - One CTA per matrix; threads = 128 = 4 warps.
//   - BLK_M = 32 fixed: each warp owns 8 contiguous m-rows of the output
//     (M_STRIPES = 1 per warp). Mirrors the m8n8k4 lane fragment layout
//     directly — no per-warp m-stripe splitting like NN's BLK_M=64.
//     For M > 32 (nw1 ∈ {44, 50}) the CTA loops over m-strips internally;
//     sB stays resident across strips so B is only loaded once per matrix.
//   - Tile rungs selected at host dispatch by N (BLK_M is fixed):
//       N <= 16     : (BLK_M, BLK_N) = (32, 16)   covers nw2 in {4, 9, 13, 16}
//       16 < N <= 32: (BLK_M, BLK_N) = (32, 32)   covers nw2 in {25, 27}
//       N >  32     : (BLK_M, BLK_N) = (32, 56)   covers nw2 in {44, 50}
//   - Inner loop: mma.sync.aligned.m8n8k4.row.col.f64 on sm_80+; per-lane
//     scalar ld.shared.f64 fragment loads (no ldmatrix until Phase 4 perf).
//   - K-tail: shmem K-pad rounded up to next multiple of 4; tail rows zero-
//     filled at load time so mma reads zeros for k >= K (bit-exact w.r.t.
//     truncating the K-sum).
//
// Shmem layout:
//   sA [K_pad x BLK_M]  K-major, M-inner   sA[k * BLK_M + m]
//   sB [K_pad x BLK_N]  K-major, N-inner   sB[k * BLK_N + n]
//   K_pad = round_up(K, 4). K-major sA matches HBM (K x M) row-major coalesced
//   reads naturally; the mma A frag load is reformulated to read the M-inner
//   stride (no ldmatrix.trans needed for Phase 3 — see plan §"TN kernel"
//   Option A).
//
// Lane fragment layout (PTX, m8n8k4.row.col.f64; PTX ISA 8.5 Table 38):
//   For lane (lr = lane >> 2 ∈ 0..7,  lc = lane & 3 ∈ 0..3):
//     A frag (8 m × 4 k row-major): rA = A_view[m=lr, k=lc]   (1 double / lane)
//     B frag (4 k × 8 n col-major): rB = B[k=lc, n=lr]        (1 double / lane)
//     D frag (8 m × 8 n row-major): rD[0] = D[lr, 2*lc],
//                                   rD[1] = D[lr, 2*lc + 1]   (2 doubles / lane)
//
//   For TN, A_view[m, k] is the (M x K) view of A^T, so A_view[m, k] is the
//   HBM element A[k, m] (K-major in HBM, lda stride). With sA stored K-major
//   M-inner, A_view[m, k] lives at sA[k * BLK_M + m] — i.e. lane (lr, lc)
//   reads sA[(k_base + lc) * BLK_M + (warp_row + lr)].
//
// FP32 + sm_70/75 FP64: forwarded to gemm_tn_vbatch_scalar_dispatch<T>.
// ============================================================================

namespace gemm_tn_v2 {

// ----------------------------------------------------------------------------
// FP64 mma kernel.
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
    static_assert(BLK_M == 32,
                  "BLK_M must be 32 (4 warps * 8 m-rows / warp = 1 m=8 stripe / warp)");
    static_assert(BLK_N % 8 == 0,
                  "BLK_N must be a multiple of 8 (mma n-frag = 8)");

    constexpr int N_FRAGS = BLK_N / 8;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int warp_row = warp_id * 8;     // 0, 8, 16, 24
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
    double* sB = sA + static_cast<size_t>(K_pad) * BLK_M;

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
        // sA[k * BLK_M + m] = (k < K && m_g < M) ? A[k*LDA + m_g] : 0.0
        // m_g = m_base + m. HBM read is K-row × M-col (consecutive lanes
        // hit consecutive m → coalesced); shmem write is K-major M-inner.
        {
            const int total = K_pad * BLK_M;
            #pragma unroll 1
            for (int idx = tid; idx < total; idx += 128)
            {
                int k = idx / BLK_M;
                int m = idx - k * BLK_M;
                int m_g = m_base + m;
                double v = 0.0;
                if (k < K && m_g < M) v = A[k * LDA + m_g];
                sA[k * BLK_M + m] = v;
            }
        }

        __syncthreads();

        // -------- Stage 2: mma inner loop. -----------------------------------
        // Per-lane register footprint:
        //   rA              : A frag, 1 double
        //   rB              : B frag (live within k_step), 1 double
        //   rC[N_FRAGS][2]  : C frag, lives across all k_steps
        double rC[N_FRAGS][2];
        #pragma unroll
        for (int nf = 0; nf < N_FRAGS; ++nf)
        {
            rC[nf][0] = 0.0;
            rC[nf][1] = 0.0;
        }

        for (int k_step = 0; k_step < K_steps; ++k_step)
        {
            const int k_base = k_step * 4;

            // A frag: lane (lr, lc) reads A_view[m=warp_row+lr, k=k_base+lc].
            // sA K-major M-inner: index = (k_base+lc)*BLK_M + (warp_row+lr).
            const double rA = sA[(k_base + lc) * BLK_M + (warp_row + lr)];

            #pragma unroll
            for (int nf = 0; nf < N_FRAGS; ++nf)
            {
                // B frag: lane (lr, lc) reads B[k=k_base+lc, n=nf*8+lr].
                // sB K-major N-inner: index = (k_base+lc)*BLK_N + (nf*8+lr).
                const double rB = sB[(k_base + lc) * BLK_N + (nf * 8 + lr)];

                mma_m8n8k4_f64(rC[nf], rA, rB);
            }
        }

        // -------- Stage 3: atomicAdd rC into C, masking m and n. -------------
        // Per-lane writes 2 C values per n_frag (D[lr, 2*lc] / D[lr, 2*lc+1]).
        const int n_left_lc  = 2 * lc;
        const int n_right_lc = n_left_lc + 1;
        const int m_g = m_base + warp_row + lr;

        if (m_g < M)
        {
            #pragma unroll
            for (int nf = 0; nf < N_FRAGS; ++nf)
            {
                const int n_left_g  = nf * 8 + n_left_lc;
                const int n_right_g = nf * 8 + n_right_lc;

                if (n_left_g  < N)
                    atomicAdd(C + m_g * LDC + n_left_g,  rC[nf][0] * alpha);
                if (n_right_g < N)
                    atomicAdd(C + m_g * LDC + n_right_g, rC[nf][1] * alpha);
            }
        }

        // sB persists across strips; sA will be overwritten by the next strip.
        if (s + 1 < n_strips) __syncthreads();
    }
}

// ----------------------------------------------------------------------------
// Host launcher for the FP64 mma kernel.
//
// Shmem opt-in: at the (32, 56) rung with K=125 (case6 bxyz=125), K_pad=128
// and the dynamic shmem hits ~88 KB, well over the 48 KB default per-block
// cap. cudaFuncSetAttribute raises the cap to 96 KB — covers our worst TN
// case (~88 KB) with margin and is within every sm_80+ opt-in limit:
//   sm_80 (A100): 163 KB    sm_86 (RTX 3090): 99 KB
//   sm_89 (RTX 4090): 99 KB sm_90 (H100):    227 KB
// 96 KB is the largest portable value that still fits sm_86's 99 KB cap
// (used for force-v2 correctness validation on the dev box). Call is
// idempotent and gated by a per-instantiation `static int` so it runs once
// per kernel symbol per process.
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
            96 * 1024);
        if (e != cudaSuccess)
        {
            fprintf(stderr,
                    "gemm_tn_v2: cudaFuncSetAttribute(MaxDynamicSharedMemorySize) "
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
    CHECK_LAST_CUDA_ERROR("gemm_tn_v2::mma_fp64_kernel launch");
}

// ----------------------------------------------------------------------------
// FP64-only v2 attempt — selects the rung by N. Returns true on dispatch.
// Mirrors nn_try_v2_'s C++14 SFINAE pattern: non-template double overload is
// the preferred candidate when T = double; the template fallback returns
// false for any other dtype so FP32 falls through to the existing scalar
// dispatcher (kept untouched in Phase 3).
// ----------------------------------------------------------------------------
inline bool tn_try_v2_(
    int m, int n, int k,
    const double* const* A_array_d, const int* lda_d,
    const double* const* B_array_d, const int* ldb_d,
    double** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream, const double* alpha)
{
    if (n <= 16)
    {
        launch_mma_fp64<32, 16>(
            m, n, k,
            A_array_d, lda_d, B_array_d, ldb_d,
            C_array_d, ldc_d, batchCount, stream, alpha);
    }
    else if (n <= 32)
    {
        launch_mma_fp64<32, 32>(
            m, n, k,
            A_array_d, lda_d, B_array_d, ldb_d,
            C_array_d, ldc_d, batchCount, stream, alpha);
    }
    else
    {
        launch_mma_fp64<32, 56>(
            m, n, k,
            A_array_d, lda_d, B_array_d, ldb_d,
            C_array_d, ldc_d, batchCount, stream, alpha);
    }
    return true;
}

template <typename T>
inline bool tn_try_v2_(
    int /*m*/, int /*n*/, int /*k*/,
    const T* const* /*A*/, const int* /*lda*/,
    const T* const* /*B*/, const int* /*ldb*/,
    T** /*C*/, const int* /*ldc*/,
    int /*batch*/, cudaStream_t /*stream*/, const T* /*alpha*/)
{
    return false;
}

}  // namespace gemm_tn_v2

template<typename T>
void gemm_tn_vbatch_v2_dispatch(
    int m, int n, int k,
    const T* const* A_array_d, const int* lda_d,
    const T* const* B_array_d, const int* ldb_d,
    T** C_array_d, const int* ldc_d,
    int batchCount, cudaStream_t stream,
    const T* alpha)
{
    if (gemm_tn_v2::tn_try_v2_(m, n, k,
                                A_array_d, lda_d, B_array_d, ldb_d,
                                C_array_d, ldc_d, batchCount, stream, alpha))
    {
        return;
    }
    // Fallback: FP32 / non-FP64 dtypes route through the scalar dispatcher.
    // Phase 3 keeps the existing FP32 TN kernel intact (HBM-bound, well-tuned).
    gemm_tn_vbatch_scalar_dispatch<T>(
        m, n, k,
        A_array_d, lda_d, B_array_d, ldb_d,
        C_array_d, ldc_d, batchCount, stream, alpha);
}

#endif // GEMM_TN_VBATCH_V2_CUH
