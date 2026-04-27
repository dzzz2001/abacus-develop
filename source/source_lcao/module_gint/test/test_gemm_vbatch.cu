// ----------------------------------------------------------------------------
// test_gemm_vbatch.cu
//
// Standalone correctness + perf microbench for the Gint batched GEMM kernels
// (gemm_nn_vbatch, gemm_tn_vbatch in dgemm_vbatch.cu). Uses a CPU-naive
// reference to validate kernel output for the row-major external contract;
// reports per-shape GFLOPS for quick iteration outside the full ABACUS SCF.
//
// External contract under test (row-major, no swap):
//   NN: C[M,N] = α * A[M,K] * B[K,N]    (atomic accumulate into C)
//   TN: C[M,N] = α * A[K,M]^T * B[K,N]  (atomic accumulate into C)
//
// Build (manual; not wired into the main ABACUS build by default):
//   nvcc -O3 -std=c++14 \
//     -gencode arch=compute_80,code=sm_80 \
//     -gencode arch=compute_86,code=sm_86 \
//     -DENABLE_LCAO -D__CUDA \
//     -I /home/dzc/abacus/abacus-gemm-opt/source \
//     /home/dzc/abacus/abacus-gemm-opt/source/source_lcao/module_gint/kernel/dgemm_vbatch.cu \
//     /home/dzc/abacus/abacus-gemm-opt/source/source_lcao/module_gint/test/test_gemm_vbatch.cu \
//     -lcudart -o test_gemm_vbatch
//
//   ./test_gemm_vbatch
//
// CMake integration: see ../test/CMakeLists.txt for the gint_vbatch_test target.
// ----------------------------------------------------------------------------

#include "../kernel/dgemm_vbatch.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t e = (call);                                              \
        if (e != cudaSuccess) {                                              \
            std::fprintf(stderr,                                             \
                         "CUDA error %s at %s:%d: %s\n",                     \
                         #call, __FILE__, __LINE__, cudaGetErrorString(e));  \
            std::exit(2);                                                    \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// CPU naive references (row-major).
// ---------------------------------------------------------------------------

static void cpu_gemm_nn_rowmajor(
    int M, int N, int K,
    const double* A, int lda,
    const double* B, int ldb,
    double* C, int ldc,
    double alpha)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += A[i*lda + k] * B[k*ldb + j];
            }
            C[i*ldc + j] += alpha * sum;
        }
    }
}

static void cpu_gemm_tn_rowmajor(
    int M, int N, int K,
    const double* A, int lda,
    const double* B, int ldb,
    double* C, int ldc,
    double alpha)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += A[k*lda + i] * B[k*ldb + j];
            }
            C[i*ldc + j] += alpha * sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

// Combined absolute + relative residual:
//   pass if |got - ref| <= atol + rtol * |ref|.
// Reported as worst-case (|got - ref| - atol) / max(|ref|, 1) so that values
// near zero don't blow up the rel scale.
static double max_normalized_residual(const std::vector<double>& got,
                                      const std::vector<double>& ref,
                                      double atol)
{
    double max_norm = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double diff = std::fabs(got[i] - ref[i]);
        const double scale = std::max(std::fabs(ref[i]), 1.0);
        const double residual = std::max(0.0, diff - atol) / scale;
        if (residual > max_norm) max_norm = residual;
    }
    return max_norm;
}

struct Stats {
    double max_rel_err = 0.0;   // normalized residual (see max_normalized_residual)
    double gflops = 0.0;
    double ms = 0.0;
};

// Random-init a host buffer in [-1, 1].
static void rand_fill(std::vector<double>& buf, std::mt19937& rng)
{
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& v : buf) v = dist(rng);
}

// ---------------------------------------------------------------------------
// One-shape NN test.
//   Allocates batchCount uniformly-shaped (M, N, K) matrices with strides
//   lda = K, ldb = N, ldc = N (no padding). Compares kernel output against
//   the CPU reference summed over all batch entries (atomicAdd writes a
//   monotonic sum which matches the reference within FP64 round-off).
// ---------------------------------------------------------------------------
static Stats run_nn_one(int M, int N, int K, int batchCount, double alpha,
                        std::mt19937& rng)
{
    const int lda = K, ldb = N, ldc = N;
    const size_t bytes_A_one = sizeof(double) * M * lda;
    const size_t bytes_B_one = sizeof(double) * K * ldb;
    const size_t bytes_C_one = sizeof(double) * M * ldc;

    // Per-batch matrices live contiguously in one big buffer.
    std::vector<double> hA(M * lda * batchCount);
    std::vector<double> hB(K * ldb * batchCount);
    std::vector<double> hC0(M * ldc * batchCount, 0.0);   // initial C value
    std::vector<double> hC_ref(M * ldc * batchCount, 0.0);
    rand_fill(hA, rng);
    rand_fill(hB, rng);

    // CPU reference: each batch entry is independent (we write to its own C
    // block). We test atomic accumulation by adding two passes with the same
    // alpha on top of an initial random C.
    std::vector<double> hC_init(hC0.size());
    rand_fill(hC_init, rng);
    std::copy(hC_init.begin(), hC_init.end(), hC0.begin());
    std::copy(hC_init.begin(), hC_init.end(), hC_ref.begin());

    for (int b = 0; b < batchCount; ++b) {
        cpu_gemm_nn_rowmajor(M, N, K,
                             hA.data() + (size_t)b * M * lda, lda,
                             hB.data() + (size_t)b * K * ldb, ldb,
                             hC_ref.data() + (size_t)b * M * ldc, ldc,
                             alpha);
    }

    // GPU buffers.
    double *dA_buf = nullptr, *dB_buf = nullptr, *dC_buf = nullptr;
    CHECK(cudaMalloc(&dA_buf, bytes_A_one * batchCount));
    CHECK(cudaMalloc(&dB_buf, bytes_B_one * batchCount));
    CHECK(cudaMalloc(&dC_buf, bytes_C_one * batchCount));
    CHECK(cudaMemcpy(dA_buf, hA.data(), bytes_A_one * batchCount,
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB_buf, hB.data(), bytes_B_one * batchCount,
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dC_buf, hC0.data(), bytes_C_one * batchCount,
                     cudaMemcpyHostToDevice));

    // Pointer arrays + stride arrays (host then device).
    std::vector<const double*> hA_ptrs(batchCount);
    std::vector<const double*> hB_ptrs(batchCount);
    std::vector<double*> hC_ptrs(batchCount);
    std::vector<int> hLda(batchCount, lda), hLdb(batchCount, ldb), hLdc(batchCount, ldc);
    for (int b = 0; b < batchCount; ++b) {
        hA_ptrs[b] = dA_buf + (size_t)b * M * lda;
        hB_ptrs[b] = dB_buf + (size_t)b * K * ldb;
        hC_ptrs[b] = dC_buf + (size_t)b * M * ldc;
    }

    const double **dA_ptrs = nullptr, **dB_ptrs = nullptr;
    double** dC_ptrs = nullptr;
    int *dLda = nullptr, *dLdb = nullptr, *dLdc = nullptr;
    CHECK(cudaMalloc(&dA_ptrs, sizeof(double*) * batchCount));
    CHECK(cudaMalloc(&dB_ptrs, sizeof(double*) * batchCount));
    CHECK(cudaMalloc(&dC_ptrs, sizeof(double*) * batchCount));
    CHECK(cudaMalloc(&dLda, sizeof(int) * batchCount));
    CHECK(cudaMalloc(&dLdb, sizeof(int) * batchCount));
    CHECK(cudaMalloc(&dLdc, sizeof(int) * batchCount));
    CHECK(cudaMemcpy(dA_ptrs, hA_ptrs.data(), sizeof(double*) * batchCount,
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB_ptrs, hB_ptrs.data(), sizeof(double*) * batchCount,
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dC_ptrs, hC_ptrs.data(), sizeof(double*) * batchCount,
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dLda, hLda.data(), sizeof(int) * batchCount,
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dLdb, hLdb.data(), sizeof(int) * batchCount,
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dLdc, hLdc.data(), sizeof(int) * batchCount,
                     cudaMemcpyHostToDevice));

    // Per-batch alpha (mirrors phi_mul_dm's is_symm pattern).
    std::vector<double> hAlpha(batchCount, alpha);
    double* dAlpha = nullptr;
    CHECK(cudaMalloc(&dAlpha, sizeof(double) * batchCount));
    CHECK(cudaMemcpy(dAlpha, hAlpha.data(), sizeof(double) * batchCount,
                     cudaMemcpyHostToDevice));

    // Warmup + timed launch.
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    gemm_nn_vbatch<double>(M, N, K, dA_ptrs, dLda, dB_ptrs, dLdb,
                           dC_ptrs, dLdc, batchCount, /*stream=*/0, dAlpha);
    CHECK(cudaDeviceSynchronize());

    constexpr int reps = 5;
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r) {
        // Reset C to baseline so each rep does the same work.
        CHECK(cudaMemcpy(dC_buf, hC0.data(), bytes_C_one * batchCount,
                         cudaMemcpyHostToDevice));
        gemm_nn_vbatch<double>(M, N, K, dA_ptrs, dLda, dB_ptrs, dLdb,
                               dC_ptrs, dLdc, batchCount, /*stream=*/0, dAlpha);
    }
    cudaEventRecord(e1);
    CHECK(cudaEventSynchronize(e1));

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    ms /= reps;

    // Copy back final result and compare.
    std::vector<double> hC_got(M * ldc * batchCount);
    CHECK(cudaMemcpy(hC_got.data(), dC_buf, bytes_C_one * batchCount,
                     cudaMemcpyDeviceToHost));

    Stats s;
    // K-element FMA reduction has accumulated absolute error ~ K * eps_d * scale.
    // Use atol = 1e-10 (well above 50 * 2.2e-16 * 100 ≈ 1e-12) plus rtol-style
    // normalization so near-zero ref values don't blow up the residual.
    const double atol = 1e-10;
    s.max_rel_err = max_normalized_residual(hC_got, hC_ref, atol);
    s.ms = ms;
    s.gflops = (2.0 * M * N * K * batchCount) / (ms * 1e6);

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(dA_buf); cudaFree(dB_buf); cudaFree(dC_buf);
    cudaFree(dA_ptrs); cudaFree(dB_ptrs); cudaFree(dC_ptrs);
    cudaFree(dLda); cudaFree(dLdb); cudaFree(dLdc);
    cudaFree(dAlpha);
    return s;
}

// ---------------------------------------------------------------------------
// One-shape TN test. A is (K x M), B is (K x N); C += α * A^T * B.
// ---------------------------------------------------------------------------
static Stats run_tn_one(int M, int N, int K, int batchCount, double alpha,
                        std::mt19937& rng)
{
    const int lda = M, ldb = N, ldc = N;
    const size_t bytes_A_one = sizeof(double) * K * lda;
    const size_t bytes_B_one = sizeof(double) * K * ldb;
    const size_t bytes_C_one = sizeof(double) * M * ldc;

    std::vector<double> hA(K * lda * batchCount);
    std::vector<double> hB(K * ldb * batchCount);
    std::vector<double> hC0(M * ldc * batchCount, 0.0);
    std::vector<double> hC_ref(M * ldc * batchCount, 0.0);
    rand_fill(hA, rng);
    rand_fill(hB, rng);

    std::vector<double> hC_init(hC0.size());
    rand_fill(hC_init, rng);
    std::copy(hC_init.begin(), hC_init.end(), hC0.begin());
    std::copy(hC_init.begin(), hC_init.end(), hC_ref.begin());

    for (int b = 0; b < batchCount; ++b) {
        cpu_gemm_tn_rowmajor(M, N, K,
                             hA.data() + (size_t)b * K * lda, lda,
                             hB.data() + (size_t)b * K * ldb, ldb,
                             hC_ref.data() + (size_t)b * M * ldc, ldc,
                             alpha);
    }

    double *dA_buf = nullptr, *dB_buf = nullptr, *dC_buf = nullptr;
    CHECK(cudaMalloc(&dA_buf, bytes_A_one * batchCount));
    CHECK(cudaMalloc(&dB_buf, bytes_B_one * batchCount));
    CHECK(cudaMalloc(&dC_buf, bytes_C_one * batchCount));
    CHECK(cudaMemcpy(dA_buf, hA.data(), bytes_A_one * batchCount,
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB_buf, hB.data(), bytes_B_one * batchCount,
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dC_buf, hC0.data(), bytes_C_one * batchCount,
                     cudaMemcpyHostToDevice));

    std::vector<const double*> hA_ptrs(batchCount);
    std::vector<const double*> hB_ptrs(batchCount);
    std::vector<double*> hC_ptrs(batchCount);
    std::vector<int> hLda(batchCount, lda), hLdb(batchCount, ldb), hLdc(batchCount, ldc);
    for (int b = 0; b < batchCount; ++b) {
        hA_ptrs[b] = dA_buf + (size_t)b * K * lda;
        hB_ptrs[b] = dB_buf + (size_t)b * K * ldb;
        hC_ptrs[b] = dC_buf + (size_t)b * M * ldc;
    }
    const double **dA_ptrs = nullptr, **dB_ptrs = nullptr;
    double** dC_ptrs = nullptr;
    int *dLda = nullptr, *dLdb = nullptr, *dLdc = nullptr;
    CHECK(cudaMalloc(&dA_ptrs, sizeof(double*) * batchCount));
    CHECK(cudaMalloc(&dB_ptrs, sizeof(double*) * batchCount));
    CHECK(cudaMalloc(&dC_ptrs, sizeof(double*) * batchCount));
    CHECK(cudaMalloc(&dLda, sizeof(int) * batchCount));
    CHECK(cudaMalloc(&dLdb, sizeof(int) * batchCount));
    CHECK(cudaMalloc(&dLdc, sizeof(int) * batchCount));
    CHECK(cudaMemcpy(dA_ptrs, hA_ptrs.data(), sizeof(double*) * batchCount,
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB_ptrs, hB_ptrs.data(), sizeof(double*) * batchCount,
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dC_ptrs, hC_ptrs.data(), sizeof(double*) * batchCount,
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dLda, hLda.data(), sizeof(int) * batchCount,
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dLdb, hLdb.data(), sizeof(int) * batchCount,
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dLdc, hLdc.data(), sizeof(int) * batchCount,
                     cudaMemcpyHostToDevice));

    std::vector<double> hAlpha(batchCount, alpha);
    double* dAlpha = nullptr;
    CHECK(cudaMalloc(&dAlpha, sizeof(double) * batchCount));
    CHECK(cudaMemcpy(dAlpha, hAlpha.data(), sizeof(double) * batchCount,
                     cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    gemm_tn_vbatch<double>(M, N, K, dA_ptrs, dLda, dB_ptrs, dLdb,
                           dC_ptrs, dLdc, batchCount, /*stream=*/0, dAlpha);
    CHECK(cudaDeviceSynchronize());

    constexpr int reps = 5;
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r) {
        CHECK(cudaMemcpy(dC_buf, hC0.data(), bytes_C_one * batchCount,
                         cudaMemcpyHostToDevice));
        gemm_tn_vbatch<double>(M, N, K, dA_ptrs, dLda, dB_ptrs, dLdb,
                               dC_ptrs, dLdc, batchCount, /*stream=*/0, dAlpha);
    }
    cudaEventRecord(e1);
    CHECK(cudaEventSynchronize(e1));

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    ms /= reps;

    std::vector<double> hC_got(M * ldc * batchCount);
    CHECK(cudaMemcpy(hC_got.data(), dC_buf, bytes_C_one * batchCount,
                     cudaMemcpyDeviceToHost));

    Stats s;
    // K-element FMA reduction has accumulated absolute error ~ K * eps_d * scale.
    // Use atol = 1e-10 (well above 50 * 2.2e-16 * 100 ≈ 1e-12) plus rtol-style
    // normalization so near-zero ref values don't blow up the residual.
    const double atol = 1e-10;
    s.max_rel_err = max_normalized_residual(hC_got, hC_ref, atol);
    s.ms = ms;
    s.gflops = (2.0 * M * N * K * batchCount) / (ms * 1e6);

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(dA_buf); cudaFree(dB_buf); cudaFree(dC_buf);
    cudaFree(dA_ptrs); cudaFree(dB_ptrs); cudaFree(dC_ptrs);
    cudaFree(dLda); cudaFree(dLdb); cudaFree(dLdc);
    cudaFree(dAlpha);
    return s;
}

// ---------------------------------------------------------------------------
// Test driver.
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    // Pass if normalized residual <= tol. The residual subtracts an
    // absolute floor (1e-10) before dividing by max(|ref|, 1), so we're
    // effectively asking: ignoring ULP-level noise, are got and ref close
    // in relative terms? 1e-12 is a clean order-of-magnitude bound for
    // FP64 GEMMs of K ≤ 125 with operands in [-1, 1].
    constexpr double tol = 1e-12;

    // Representative shapes drawn from the ABACUS Gint workload (108-atom
    // suite, bxyz ∈ {27, 64, 125}, nw ∈ {4, 9, 13, 25, 27, 44, 50}).
    const std::vector<int> M_nn = {27, 64, 125};
    const std::vector<int> N_nn = {9, 13, 25, 44, 50};
    const std::vector<int> K_nn = {9, 13, 25, 44, 50};

    const std::vector<int> M_tn = {9, 13, 25, 44, 50};
    const std::vector<int> N_tn = {9, 13, 25, 44, 50};
    const std::vector<int> K_tn = {27, 64, 125};

    const int batchCount = 32;
    const double alpha = 2.0;
    std::mt19937 rng(0xC0FFEEu);

    int dev = 0;
    cudaDeviceProp props;
    CHECK(cudaGetDevice(&dev));
    CHECK(cudaGetDeviceProperties(&props, dev));
    std::printf("Device 0: %s (sm_%d%d)\n", props.name, props.major, props.minor);
    std::printf("Tolerance (max-rel-error) = %.1e\n", tol);
    std::printf("batchCount = %d, alpha = %g\n\n", batchCount, alpha);

    int n_total = 0, n_fail = 0;

    std::printf("=== NN: C[M,N] += alpha * A[M,K] * B[K,N] ===\n");
    std::printf("%4s %4s %4s %12s %10s %10s\n",
                "M", "N", "K", "max_rel_err", "ms/iter", "GFLOP/s");
    for (int M : M_nn) for (int N : N_nn) for (int K : K_nn) {
        Stats s = run_nn_one(M, N, K, batchCount, alpha, rng);
        const bool ok = s.max_rel_err <= tol;
        std::printf("%4d %4d %4d %12.2e %10.4f %10.2f%s\n",
                    M, N, K, s.max_rel_err, s.ms, s.gflops,
                    ok ? "" : "  FAIL");
        ++n_total;
        if (!ok) ++n_fail;
    }

    std::printf("\n=== TN: C[M,N] += alpha * A[K,M]^T * B[K,N] ===\n");
    std::printf("%4s %4s %4s %12s %10s %10s\n",
                "M", "N", "K", "max_rel_err", "ms/iter", "GFLOP/s");
    for (int M : M_tn) for (int N : N_tn) for (int K : K_tn) {
        Stats s = run_tn_one(M, N, K, batchCount, alpha, rng);
        const bool ok = s.max_rel_err <= tol;
        std::printf("%4d %4d %4d %12.2e %10.4f %10.2f%s\n",
                    M, N, K, s.max_rel_err, s.ms, s.gflops,
                    ok ? "" : "  FAIL");
        ++n_total;
        if (!ok) ++n_fail;
    }

    std::printf("\nResult: %d / %d shapes passed\n", n_total - n_fail, n_total);
    return n_fail == 0 ? 0 : 1;
}
