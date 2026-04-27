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
// arch supports the FP64 mma PTX instruction (sm_80+). HW acceleration is
// only present on sm_80 and sm_90; on sm_86 / sm_89 the instruction *decodes*
// but issues at scalar FP64 rate (1/64 FP32). Production routing of FP64 to
// v2 happens only on sm_80/90 — see dgemm_vbatch.cu's fp64_use_v2_kernel().
// The macro is widened to all sm_80+ so the v2 kernel is bit-correct (just
// slow) when force-routed to consumer Ampere/Ada for correctness validation.
// ----------------------------------------------------------------------------

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
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
// FP64 mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64
// Per warp: D[8,8] += A[8,4] * B[4,8]
// Per lane register footprint: A=1 double, B=1 double, C/D=2 doubles.
//
// Why m8n8k4 (and not m16n8k4 / m16n8k8): m8n8k4 is the only FP64 mma shape
// supported on sm_80 (A100). m16n8k4, m16n8k8, m16n8k16 with .f64 all require
// .target sm_90 or higher — verified empirically: ptxas rejects them with
// "Feature '.m16n8k* with double types' requires .target sm_90 or higher"
// when emitting compute_80 PTX. m8n8k4 still saturates the FP64 TC at
// 19.5 TFLOPS on sm_80; the only cost vs m16n8k8 is a 2x in issue count,
// which is irrelevant for our HBM-bound workload.
//
// A sm_90 specialization to m16n8k8 (4× fewer issues per K-band) is deferred
// to a future perf phase.
//
// PTX form per PTX ISA 8.5 Section 9.7.16.5, Table 38 (m8n8k4 with .f64).
// ----------------------------------------------------------------------------
__device__ __forceinline__
void mma_m8n8k4_f64(double* acc, double a, double b)
{
#if GEMM_HAS_FP64_TC
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
        "{%0,  %1}, "
        "{%2}, "
        "{%3}, "
        "{%0,  %1};\n"
        : "+d"(acc[0]), "+d"(acc[1])
        : "d"(a),
          "d"(b));
#else
    // Fallback for non-TC arches (or host compilation): unreachable at
    // runtime when the host dispatcher routes correctly. Phase 2 ships this
    // as a no-op since FP32 / sm_70/75 fall through to the scalar dispatch.
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
