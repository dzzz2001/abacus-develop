#include "gemm_tn_vbatch_scalar.cuh"
#include "gemm_nn_vbatch_scalar.cuh"
#include "gemm_tn_vbatch_v2.cuh"
#include "gemm_nn_vbatch_v2.cuh"
#include "dgemm_vbatch.h"
#include "source_base/module_device/device.h"

#include <cstdlib>  // std::getenv

// ----------------------------------------------------------------------------
// SM-arch dispatch (Phase 1 scaffolding, Phase 2+ active)
// ----------------------------------------------------------------------------
// The v2 kernels (gemm_*_vbatch_v2.cuh) target A100 / H100 specifically: they
// rely on hardware FP64 mma (sm_80, sm_90). On sm_70 / sm_75 the mma.f64
// instruction does not exist; on sm_86 / sm_89 it decodes but throttles to
// scalar FP64 rate (consumer Ampere / Ada have FP64 = 1/64 FP32). On those
// arches the existing scalar tile-ladder kernel is faster, so we route FP64
// GEMMs through the scalar dispatch.
//
// FP32 always goes through v2 — it has no FP32 TC story (TF32 is a different
// precision), so v2's scalar FMA inner loop is the natural path; FP32 callers
// also benefit from the cleaner row-major contract once Phase 2 lands. In
// Phase 1 the v2 stub forwards to the same scalar ladder, so behavior is
// unchanged.
//
// Detection runs once per process via std::call_once: properties of GPU 0 at
// the time of the first GEMM call. Multi-GPU runs that mix sm versions are
// not supported (the existing kernel suite assumes one CUDA device).
// ----------------------------------------------------------------------------

namespace {

bool detect_fp64_use_v2()
{
    // Test override: ABACUS_GEMM_FORCE_V2_FP64=1 routes FP64 to v2 on any
    // sm_80+ arch. mma.f64 PTX decodes on consumer Ampere/Ada at scalar
    // FP64 rate, so the kernel is bit-correct (just slow) — used to validate
    // the v2 kernel locally on RTX 3090 (sm_86) before A100 perf runs.
    if (const char* env = std::getenv("ABACUS_GEMM_FORCE_V2_FP64"))
    {
        if (env[0] == '1') { return true; }
    }

    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, dev);
    const int sm = props.major * 10 + props.minor;
    return (sm == 80 || sm == 90);
}

bool fp64_use_v2_kernel()
{
    static const bool cached = detect_fp64_use_v2();
    return cached;
}

// Per-dtype routing predicate. C++14 specialization (avoids `if constexpr`).
//   double : runtime arch detection
//   float  : v2 always (no FP32 TC, but v2 has scalar FMA inner loop)
template<typename T>
inline bool gemm_use_v2_for_dtype();

template<>
inline bool gemm_use_v2_for_dtype<double>() { return fp64_use_v2_kernel(); }

template<>
inline bool gemm_use_v2_for_dtype<float>() { return true; }

}  // namespace

// ----------------------------------------------------------------------------
// Public dispatch — see phi_operator_gpu.cu (lines 342, 449) for callers.
// Shape-exact: the caller buckets atom pairs by (nw1, nw2) so every batch
// item has the same (m, n, k); both v2 and scalar paths assume this.
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
    const bool use_v2 = gemm_use_v2_for_dtype<T>();

    if (use_v2) {
        gemm_nn_vbatch_v2_dispatch<T>(
            m, n, k,
            A_array_d, lda_d, B_array_d, ldb_d,
            C_array_d, ldc_d, batchCount, stream, alpha);
    } else {
        gemm_nn_vbatch_scalar_dispatch<T>(
            m, n, k,
            A_array_d, lda_d, B_array_d, ldb_d,
            C_array_d, ldc_d, batchCount, stream, alpha);
    }
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
    const bool use_v2 = gemm_use_v2_for_dtype<T>();

    if (use_v2) {
        gemm_tn_vbatch_v2_dispatch<T>(
            m, n, k,
            A_array_d, lda_d, B_array_d, ldb_d,
            C_array_d, ldc_d, batchCount, stream, alpha);
    } else {
        gemm_tn_vbatch_scalar_dispatch<T>(
            m, n, k,
            A_array_d, lda_d, B_array_d, ldb_d,
            C_array_d, ldc_d, batchCount, stream, alpha);
    }
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
