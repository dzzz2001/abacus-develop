# A100-targeted batched FP64 GEMM redesign for ABACUS Gint

> Authoritative copy of the implementation plan. Mirror of
> `~/.claude/plans/1-2-gpu-tensor-luminous-reef.md`. See
> `docs/gemm_vbatch_tuning_guide.md` for the legacy kernel anatomy.

## Context

The Gint batched-GEMM kernels in
`source/source_lcao/module_gint/kernel/{gemm_nn_vbatch,gemm_tn_vbatch}.cuh`
were designed for general batched GEMM. For the Gint workload (uniform-shape
batches with M=bxyz∈[27,125], N=nw2∈[7,50], K=nw1∈[7,50] for NN; M,N small
and K=bxyz large for TN) they have three structural problems:

1. **Multi-K-tile blocking is unnecessary.** The kernels stream K in
   `BLK_K`-sized tiles with double-buffered prefetch and inter-tile
   `__syncthreads()`. For these sizes the small operand (B in NN,
   ≤50×50=20 KB FP64; both operands in TN at ≤100 KB combined) fits fully
   in shmem in one shot — the K-tile loop and its sync overhead add no value.
2. **No FP64 Tensor Cores.** A100 has `mma.sync.aligned.m16n8k8.row.col.f64`
   (19.5 TFLOPS peak vs 9.7 TFLOPS scalar). Inner loop is scalar FMA only.
3. **Column-major-native indexing with a launcher swap.** Kernels at
   `gemm_nn_vbatch.cuh:486` and `gemm_tn_vbatch.cuh:489` swap A↔B and (m,n)
   to expose a row-major contract on top of column-major-native code. No
   physical conversion happens, but the layer is opaque — the shmem layout
   convention doesn't match the natural mma `row.col` fragment layout, and
   the dataflow is hard to reason about.

This plan introduces an A100-targeted kernel that is row-major-native,
shmem-resident for the small operand, and uses FP64 mma on sm_80/90.
Existing scalar kernel is preserved as fallback for FP64 on V100/T4/3090/Ada
(no regression on non-A100 builds). Realistic perf goal: **+10-30% A100
SCF wall time on `cal_gint_vl` and `cal_gint_rho`**.

Dev box is RTX 3090 (sm_86, no FP64 TC hardware) — kernel correctness is
validated locally, perf is benchmarked on A100 by the user.

## Design overview

- **One CTA per matrix.** With ~500-2000 matrices/call and 108 SMs on A100,
  this gives 5-20× over-subscription — enough for HBM-latency hiding without
  multi-CTA output tiling and without atomic K-reduction.
- **Shmem-resident small operand.**
  - NN: B (K×N, ≤20 KB) fully in shmem; A (M×K, ≤50 KB) tiled in M (1-2
    cp.async batches per CTA).
  - TN: both A (K×M) and B (K×N) fully in shmem (≤100 KB combined; needs
    `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, 110*1024)` opt-in
    once via `std::call_once`).
- **Inner loop dispatch (compile-time):**
  - T=double on sm_80/90: `mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64`
  - T=double on other archs: scalar FMA (path unreached at runtime since
    host dispatcher routes non-sm_80/90 FP64 to the existing scalar kernel)
  - T=float, all archs: scalar FMA (no FP32 fallback path)
- **Native row-major contract:** kernel signature accepts row-major
  A `[i*lda+j]`, B `[i*ldb+j]`, C `[i*ldc+j]`. The bytes read/written are
  identical to today (verified: column-major-with-swap and row-major-direct
  produce the same byte access pattern for ABACUS's phi/dm/HR layout).
  No call-site changes in `phi_operator_gpu.cu`.
- **Atomic accumulation preserved.** `atomicAdd(C, rC * alpha)` with
  per-batch alpha — required by `phi_mul_dm`'s `is_symm` semantics
  (`phi_operator_gpu.cu:425-428`) and cross-call accumulation in
  `phi_mul_phi`.
- **Bucketed dispatch unchanged.** `phi_operator_gpu.cu`'s shape-exact
  bucketing by (nw1, nw2) (lines 277-302, 373-398) stays. New kernels still
  take uniform (M, N, K) per launch. (Bucket fusion deferred to a possible
  future Phase 5.)

## NN kernel

**File:** `source/source_lcao/module_gint/kernel/gemm_nn_vbatch_v2.cuh`

Contract — row-major:
- A is `(M × K)` at `A[i*lda+j]`, lda = phi_len_mgrid (caller-provided).
- B is `(K × N)` at `B[i*ldb+j]`, ldb = nw2.
- C is `(M × N)` at `C[i*ldc+j]`, ldc = phi_len_mgrid. Atomic accumulate
  with per-batch alpha.

Tile rungs (selection by N at host dispatch):
- `(BLK_M, BLK_N) = (64, 16)` for N ≤ 16 (covers nw2 ∈ {4,9,13,16}).
- `(BLK_M, BLK_N) = (64, 56)` for 16 < N ≤ 56 (covers nw2 ∈ {25,27,44,50}).

BLK_M is fixed at 64 — with one CTA per matrix and bxyz ∈ [27,125], this
covers most matrices in 1 M-strip (M ≤ 64) or 2 M-strips (M > 64). Mask
waste only affects the C atomic store, not register footprint (the K-loop
runs on the actual K, not BLK_K).

Threads: 128/CTA = 4 warps, each owning a (16 × BLK_N) row band of the
output. mma fragments: 1 frag in M (m=16), `BLK_N/8` frags in N.

Shmem:
- `sB[round_up(K,8) × BLK_N]` row-major in K, N-major within row.
- `sA[BLK_M × round_up(K,8)]` M-major, K-padded with shmem-bank PAD.
- (64, 16) at K=16: ~10 KB total. (64, 56) at K=50→56: ~53 KB. Both fit
  without opt-in (A100 default 100 KB/SM dynamic shmem).

cp.async load:
- One `cp.async.commit_group` per operand, fanned across 128 threads with
  16-byte (`cp.async.cg.shared.global`) per-thread copies.
- `cp.async.wait_group<0>` + `__syncthreads()` once.

Inner loop (sketch):
```cpp
for (int k_step = 0; k_step < (K+7)/8; ++k_step) {
    load_A_frag(sA, rA, lane_id, k_step);     // 4 doubles/lane (m16 × k8)
    #pragma unroll
    for (int nf = 0; nf < N_FRAGS; ++nf)
        load_B_frag(sB, rB[nf], lane_id, k_step, nf); // 2 doubles/lane (k8 × n8)

    #pragma unroll
    for (int nf = 0; nf < N_FRAGS; ++nf) {
      #if __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ != 860 && __CUDA_ARCH__ != 890
        if constexpr (std::is_same_v<T, double>)
            mma_m16n8k8_f64(rC[nf], rA, rB[nf]);
        else
            scalar_fma(rC[nf], rA, rB[nf]);
      #else
        scalar_fma(rC[nf], rA, rB[nf]);
      #endif
    }
}
```

Edge handling:
- K-tail: pad shmem K to next multiple of 8, zero-fill via cp.async
  predication (`src-bytes=0` for OOB lanes). Tail mma instructions multiply
  by zero — bit-exact w.r.t. truncating the K-sum.
- M, N: pad shmem to next multiple of 16/8; mask the C store
  `if (coord_m < M && coord_n < N) atomicAdd(...)`.

C output (row-major):
```cpp
T* C_addr = C + coord_m * LDC + coord_n;
atomicAdd(C_addr, rC[m][n] * alpha);
```
(Note the row-major offset — `coord_m * LDC + coord_n` — vs the current
column-major form `coord_dCn * LDC + coord_dCm` at
`gemm_nn_vbatch.cuh:307`.)

## TN kernel

**File:** `source/source_lcao/module_gint/kernel/gemm_tn_vbatch_v2.cuh`

Contract — row-major: `C = α A^T B + C`
- A `(K × M)` row-major, lda = phi_len_mgrid.
- B `(K × N)` row-major, ldb = phi_len_mgrid.
- C `(M × N)` row-major, ldc = nw2. Atomic accumulate.

Tile rungs (selection by max(M,N)):
- `(M, N) = (16, 16)` for both ≤ 16.
- `(M, N) = (32, 32)` for both ≤ 32.
- `(M, N) = (48, 56)` for either > 32 (covers nw=44, 50).

Threads: 128/CTA = 4 warps. Output decomposed into `m16n8k8` mma tiles.

Shmem:
- `sA[round_up(K,8) × BLK_M]` K-major (matches HBM source for cp.async
  coalescing) + PAD in M for ldmatrix-trans bank-conflict avoidance.
- `sB[round_up(K,8) × BLK_N]` K-major (matches mma B fragment layout
  natively).
- (16,16) at K=64: ~4 KB. (48,56) at K=128: ~104 KB → **needs opt-in**.

A is K-major in shmem but mma A fragment expects M-major. Two options:
- **Option A (ship-first):** scalar transposed loads from shmem to
  registers — explicit per-lane index arithmetic.
- **Option B (Phase 4 perf, optional):** `ldmatrix.sync.aligned.x4.trans.shared.b16`
  — load M-major fragment in one warp instruction, treating doubles as
  4 contiguous halfwords.

cp.async + inner loop pattern matches NN. Phase 3 ships single-stage
cp.async; Phase 4 promotes K ≥ 64 to two-stage ping-pong (50% load-latency
hide); K ≤ 32 stays single-stage.

## Architecture dispatch

Compile-time guard inside the new kernel body:
```cpp
#if __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ != 860 && __CUDA_ARCH__ != 890
  // mma.f64 path
#else
  // scalar FMA fallback
#endif
```

Run-time dispatch in `dgemm_vbatch.cu` (host-side, via `std::call_once` per
process):
```cpp
int sm = props.major * 10 + props.minor;
bool fp64_use_v2 = (sm == 80 || sm == 90);
```

Routing table:
| T      | sm 70 | sm 75 | sm 80 | sm 86 | sm 89 | sm 90 |
|--------|-------|-------|-------|-------|-------|-------|
| double | scalar| scalar| **v2 (mma)** | scalar | scalar | **v2 (mma)** |
| float  | **v2 (FMA)** | **v2 (FMA)** | **v2 (FMA)** | **v2 (FMA)** | **v2 (FMA)** | **v2 (FMA)** |

FP32 unconditionally uses the new kernel — its scalar FMA inner loop is
structurally similar to the existing kernel's, no perf regression
expected, and FP32 callers benefit from the cleaner row-major contract.

FP64 on sm_70/75/86/89 stays on the existing scalar kernel — preserves
the V100/3090 LDS-bound tuning already in the tree.

## File layout

New files:
- `source/source_lcao/module_gint/kernel/gemm_nn_vbatch_v2.cuh` — NN kernel
  + `vbatched_gemm_nn_v2_impl<T, BLK_M, BLK_N, ...>` host launcher.
- `source/source_lcao/module_gint/kernel/gemm_tn_vbatch_v2.cuh` — TN kernel
  + launcher.
- `source/source_lcao/module_gint/kernel/gemm_mma_helpers.cuh` — shared:
  `mma_m16n8k8_f64()` PTX wrapper, `cp_async_16B()` wrapper,
  `load_*_frag()` scalar shmem loaders.
- `source/source_lcao/module_gint/test/test_gemm_vbatch.cu` — microbench +
  cuBLAS reference (Phase 1).

Renamed (no behavior change):
- `gemm_nn_vbatch.cuh` → `gemm_nn_vbatch_scalar.cuh`
- `gemm_tn_vbatch.cuh` → `gemm_tn_vbatch_scalar.cuh`
- Update includes in `dgemm_vbatch.cu`.

Modified:
- `dgemm_vbatch.cu` — add SM detection static + dual-arm dispatch (FP64
  v2-vs-scalar by SM; FP32 always v2). Existing big-tile `try_big_tile_`
  path stays reachable on the FP64 scalar arm.
- `source/source_lcao/module_gint/test/CMakeLists.txt` — wire the
  microbench.

Unchanged:
- `phi_operator_gpu.cu` — call sites (lines 342, 449) pass the same args
  to `gemm_*_vbatch<T>`. The new kernel reads them as row-major (matching
  what's actually being passed); byte access pattern is identical to today.
- `dgemm_vbatch.h` — public signatures unchanged.

## Verification

**Microbench (Phase 1 deliverable):**
- `test_gemm_vbatch.cu` allocates uniform-shape batches with random data;
  computes reference C via `cublasDgemmStridedBatched` (cuBLAS already
  available via `source/source_base/module_container/base/third_party/cublas.h`);
  calls the new kernel and asserts max-rel-error < 1e-12.
- Sweeps NN shapes M ∈ {27,48,64,80,100,125} × N,K ∈ {4,9,13,25,27,44,50};
  same for TN. Reports GFLOPS per shape.
- ctest target, runnable in seconds — no full ABACUS SCF needed.

**Correctness (3090, dev box):**
- Microbench passes for all shape combinations, including odd K-tails
  (K=7, 13, 27, 50).
- Run `~/abacus/test-examples/ABACUS-test/LCAO_test/gemm_vbatch_bench/run_bench.sh`,
  diff `OUT.test/running_scf.log` energies vs `OUT.baseline/running_scf.log`
  for cases 1, 3, 6 (different bxyz). Match to per-case noise floor (~1e-8).

**Performance (A100, user-arranged):**
- Same `run_bench.sh` on A100. Compare `cal_gint_vl` and `cal_gint_rho`
  per case. Target: +10-30% on FP64 paths.
- `nsys` profile: tensor-pipe utilization > 50% on dominant tiles;
  HBM-compute overlap visible on TN K=125.
- Update `~/abacus/test-examples/ABACUS-test/LCAO_test/gemm_vbatch_bench/REPORT.md`.

## Phased implementation (1-4 only; 5-6 deferred)

### Phase 1 — scaffolding + microbench (~0.5 day) — **COMPLETE**
- Rename `gemm_{nn,tn}_vbatch.cuh` → `_scalar.cuh`; update includes.
- Stub `gemm_{nn,tn}_vbatch_v2.cuh` whose impl wrappers fall through to
  scalar (no behavior change yet).
- Skeleton `gemm_mma_helpers.cuh`.
- Wire SM detection static + dispatch in `dgemm_vbatch.cu`.
- New microbench `test_gemm_vbatch.cu` + CMakeLists wiring.

Verify: build passes on sm_86; microbench runs against fall-through;
existing `run_bench.sh` reproduces baseline numbers.

### Phase 2 — NN v2 kernel (~2 days)
- Implement `gemm_nn_vbatch_v2.cuh` per Section "NN kernel". Single rung
  `(64, 16)` first; verify against microbench cuBLAS reference; add
  `(64, 56)`.
- Wire FP32 path (unconditional) and FP64 path (sm_80/90 only).

Verify on 3090: microbench all shapes; `run_bench.sh` energies match;
FP32 timing not regressed; FP64 timing on 3090 unchanged (still scalar
fallback).

A100 perf check (user): bench `cal_gint_rho` improvement.

### Phase 3 — TN v2 kernel single-stage (~2 days)
- Implement `gemm_tn_vbatch_v2.cuh` per Section "TN kernel" with
  single-stage cp.async.
- Add `cudaFuncSetAttribute` opt-in (one-time `std::call_once`) for the
  `(48, 56, K≥64)` shmem case.
- Three rungs.

Verify: same as Phase 2 but for `cal_gint_vl`.

### Phase 4 — TN two-stage cp.async (~1 day, A100-only optimization)
- Promote TN to two-stage cp.async ping-pong for K ≥ 64 in
  `gemm_tn_vbatch_v2.cuh`. K ≤ 32 stays single-stage.

Verify on A100: nsys shows HBM-compute overlap; wall-time improves on
case 6 (K=125).

## Risks and mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| HBM-bound nature limits TC speedup | High | Realistic gain +10-30% (not 2×). Set success bar at +10% on A100. |
| sm_86 dev box can't validate perf | High | Correctness on 3090 (microbench + run_bench energies); A100 perf via user. |
| Removing swap dance breaks corner cases | Med | Microbench includes odd K-tails (7, 13, 27, 50) and asymmetric shapes; cuBLAS reference catches any indexing bug. |
| `cudaFuncSetAttribute` opt-in startup cost | Low | Once per kernel symbol via `std::call_once`. |
| FP32 perf regression from new kernel scalar FMA | Med | Inner loop structurally similar to existing scalar kernel; HBM-bound ⇒ inner-loop micro-arch doesn't matter. If observed > 5%, expand to keep FP32 on existing scalar kernel. |
| ABACUS V100 builds | Low | Existing scalar kernel preserved as fallback; compile-time `__CUDA_ARCH__` guard. |
| ldmatrix path complexity for FP64 | Low | Phase 3 ships scalar `ld.shared.f64` per-lane (proven by `papers/batched_gemm_1d_double_mma_128.cu` lines 89-102); ldmatrix is a Phase 4+ perf optimization, not blocking. |

## Critical files

- `source/source_lcao/module_gint/kernel/dgemm_vbatch.cu` — central dispatch, modified.
- `source/source_lcao/module_gint/kernel/gemm_nn_vbatch.cuh` — to rename `_scalar.cuh`. **(done)**
- `source/source_lcao/module_gint/kernel/gemm_tn_vbatch.cuh` — to rename `_scalar.cuh`. **(done)**
- `source/source_lcao/module_gint/kernel/phi_operator_gpu.cu` — caller, unchanged (call sites at lines 342, 449).
- `source/source_lcao/module_gint/kernel/gint_helper.cuh` — `ceil_div`, `gemm_vec_traits<T>` (reused for FP32 scalar FMA path).
- `papers/batched_gemm_1d_double_mma_128.cu` — reference for `mma.m16n8k8.f64` PTX form (lines 46-57) and lane fragment layout (lines 89-102).
- `source/source_base/module_container/base/third_party/cublas.h` — `cublasDgemmStridedBatched` for microbench reference.
- `CMakeLists.txt:417-426` — CUDA arch list (sm_75, 80, 86, 89, 90 — no change needed).

---

## Phase 1 status snapshot (2026-04-27)

Done:
- Renames + include-guard updates: `gemm_{nn,tn}_vbatch.cuh` → `_scalar.cuh`.
- Scalar tile-ladder dispatcher extracted into `gemm_{nn,tn}_vbatch_scalar.cuh`
  as `gemm_{nn,tn}_vbatch_scalar_dispatch<T>` so v2 stubs can forward.
- Stub `gemm_{nn,tn}_vbatch_v2.cuh` with `*_v2_dispatch<T>` forwarding to
  scalar (no behavior change yet).
- `gemm_mma_helpers.cuh` skeleton: `mma_m16n8k8_f64`, `cp_async_16B`,
  commit/wait barriers, `GEMM_HAS_FP64_TC` / `GEMM_HAS_CP_ASYNC` macros.
- `dgemm_vbatch.cu` rewritten: SM detection static (`fp64_use_v2_kernel()`),
  per-dtype routing predicate (`gemm_use_v2_for_dtype<T>()`), thin dispatch
  to v2 vs scalar arm. C++14 (no `if constexpr`).
- Microbench `test_gemm_vbatch.cu` + CMakeLists target
  `gint_vbatch_microbench` (USE_CUDA-gated, BUILD_TESTING-independent).

Verified on RTX 3090 (sm_86):
- Full ABACUS build passes.
- Microbench: 150/150 shape sweeps pass with combined atol=1e-10 + rtol=1e-12.
- ABACUS case1 (FP32, bxyz=27): rc=0; kernel timings 1.46×/1.50×/1.02× match
  the prior optimized test build's REPORT.md numbers (1.56×/1.40×/1.02×).
  Energy delta from baseline ~3.5e-3 eV is within FP32 GPU-atomic
  non-determinism (`gint_precision single` exercises the FP32 kernel path,
  which routes through the v2 stub then forwards to scalar).
