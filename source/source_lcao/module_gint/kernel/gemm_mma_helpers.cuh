#ifndef GEMM_MMA_HELPERS_CUH
#define GEMM_MMA_HELPERS_CUH
#include <cuda_runtime.h>
#include <cstdint>

// ----------------------------------------------------------------------------
// Helpers shared by the v2 (A100-targeted) Gint GEMM kernels.
//
// Three primitives, all `__device__` and `__forceinline__`:
//   1. mma_m16n8k8_f64        : FP64 Tensor-Core mma (sm_80+ HW-accelerated).
//   2. cp_async_16B           : async 16-byte global -> shared copy (sm_80+).
//   3. cp_async_commit / wait : sync barriers for the async pipeline.
//
// Architecture guard: GEMM_HAS_FP64_TC is 1 when the current device-code
// arch has hardware FP64 mma (sm_80, sm_90). On sm_86 / sm_89 the mma.f64
// instruction *decodes* but issues at scalar FP64 rate (= no benefit, plus
// extra register pressure), so we treat them as "no TC" and route to the
// scalar fallback at the host dispatcher level.
// ----------------------------------------------------------------------------

#if defined(__CUDA_ARCH__) \
    && __CUDA_ARCH__ >= 800 \
    && __CUDA_ARCH__ != 860 \
    && __CUDA_ARCH__ != 890
  #define GEMM_HAS_FP64_TC 1
#else
  #define GEMM_HAS_FP64_TC 0
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  #define GEMM_HAS_CP_ASYNC 1
#else
  #define GEMM_HAS_CP_ASYNC 0
#endif

// ----------------------------------------------------------------------------
// FP64 mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64
// Per warp: D[16,8] += A[16,8] * B[8,8]
// Per lane register footprint: A=4 doubles, B=2 doubles, C/D=4 doubles.
// PTX form lifted verbatim from papers/batched_gemm_1d_double_mma_128.cu
// (lines 46-57). Signature mirrors that PoC's mma_m16n8k8 helper.
// ----------------------------------------------------------------------------
__device__ __forceinline__
void mma_m16n8k8_f64(double* acc, const double* a, const double* b)
{
#if GEMM_HAS_FP64_TC
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64"
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%0,  %1,  %2,  %3};\n"
        : "+d"(acc[0]), "+d"(acc[1]), "+d"(acc[2]), "+d"(acc[3])
        : "d"(a[0]), "d"(a[1]), "d"(a[2]), "d"(a[3]),
          "d"(b[0]), "d"(b[1]));
#else
    // Fallback for non-TC arches (or host compilation): scalar FMA over the
    // m16n8k8 = 128-FMA chunk. Lane-fragment layout is the same; this body is
    // unreachable at runtime when the host dispatcher routes correctly.
    // Implementation deferred to Phase 2 when fragment loaders are in place.
    (void)acc; (void)a; (void)b;
#endif
}

// ----------------------------------------------------------------------------
// cp.async.cg.shared.global  -- async 16-byte global -> shared copy (Ampere+).
// Stages a 16B chunk into smem_addr; commit/wait barriers synchronize groups.
// On pre-Ampere arches, falls back to a synchronous ld.global + st.shared.
// ----------------------------------------------------------------------------
__device__ __forceinline__
void cp_async_16B(void* smem_dst, const void* gmem_src, bool zero_fill = false)
{
#if GEMM_HAS_CP_ASYNC
    uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    if (zero_fill) {
        // src-bytes = 0 zero-fills the destination per the cp.async predication
        // semantics; used for K-tail / M-tail OOB lanes.
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16, 0;\n"
            :: "r"(smem_int), "l"(gmem_src));
    } else {
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "r"(smem_int), "l"(gmem_src));
    }
#else
    // Synchronous fallback (sm < 80). Two doubles per 16B chunk.
    const double2* src = reinterpret_cast<const double2*>(gmem_src);
    double2* dst = reinterpret_cast<double2*>(smem_dst);
    *dst = zero_fill ? double2{0.0, 0.0} : *src;
#endif
}

__device__ __forceinline__
void cp_async_commit_group()
{
#if GEMM_HAS_CP_ASYNC
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__
void cp_async_wait_group()
{
#if GEMM_HAS_CP_ASYNC
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

#endif // GEMM_MMA_HELPERS_CUH
