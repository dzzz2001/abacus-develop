# `gemm_*_vbatch` optimization — progress

Working notes for the `gemm-opt` branch. Tracks what has shipped, what is in
flight, and what is queued.

## Done

### 1. Shape-based dispatch refactor (this commit)

**Goal:** decouple the kernel-template selection in `gemm_{nn,tn}_vbatch`
from the caller's `bucket_id`, so that the wrapper picks templates from
`(max_m, max_n, max_k)` and the caller no longer has to know which template
maps to which bucket. Per the agreed scoping, this is a refactor only — it
deliberately does not change perf characteristics on the cases that hit
`bxyz ≤ 64`.

**Changes:**

- `source/source_lcao/module_gint/kernel/dgemm_vbatch.h` — dropped
  `bucket_id` from both prototypes; rewrote the doc comment to describe
  shape-based dispatch.
- `source/source_lcao/module_gint/kernel/dgemm_vbatch.cu` — replaced the
  `if (bucket_id == 0) {...}` branch with a 3-tier shape-based dispatch:
  - **NN** keyed on `max_n` (nw2) then `max_m` (bxyz):
    - `max_n ≤ 8` → `BLK 8x64x16`
    - `max_n > 8 ∧ max_m ≤ 64` → `BLK 16x64x16`
    - `max_n > 8 ∧ max_m  > 64` → **NEW** `BLK 16x128x16` (covers `bxyz=125`
      in a single N-tile)
  - **TN** keyed on `max(max_m, max_n)` (max nw) then `max_k` (bxyz):
    - `max(m,n) ≤ 8` → `BLK 8x8x32`
    - `max(m,n) > 8 ∧ max_k ≤ 64` → `BLK 16x16x32`
    - `max(m,n) > 8 ∧ max_k  > 64` → **NEW** `BLK 16x16x64` (halves K-iter
      count at `bxyz=125`)
  - Asymmetric-pair safety: the new TN tier varies BLK_K only — BLK_M /
    BLK_N stay at 16, so TM-O / TM-Li pairs (13×27, 7×27) are not regressed.
- `source/source_lcao/module_gint/kernel/phi_operator_gpu.cu` — dropped the
  `b` argument from both `gemm_{tn,nn}_vbatch<Real>(...)` call sites in
  `phi_mul_phi` and `phi_mul_dm`. `gemm_bucket_of`, the DNF partition, and
  the per-bucket loop are unchanged — the caller still feeds shape-
  homogeneous sub-batches; the wrapper independently arrives at the same
  template choice.

**Verification:**

| check | result |
|---|---|
| `cmake --build build-test -j` | clean, no template warnings |
| `git grep -n bucket_id source/source_lcao/module_gint/kernel/` | 0 hits |
| Energy diff vs baseline (cases 1, 3, 6) | ≤ 2 e-9 Ha (atomicAdd reorder noise) |
| Cases 1–4 perf vs old test build | within ±1 % (same templates picked) |
| Cases 5–6 perf vs old test build | within ±1 % (new tier-2 templates land in noise) |

**Benchmark snapshot** (full table in
`~/abacus/test-examples/ABACUS-test/LCAO_test/gemm_vbatch_bench/REPORT.md`):

| timer            | best  | worst   | uniform direction? |
|------------------|------:|--------:|--------------------|
| `cal_gint_vl`    | +5.3 %| +2.9 %  | yes — all 6 cases improve |
| `cal_gint_rho`   | +4.5 %| **−34.2 %** | no — bxyz-shape sensitive |
| `cal_gint_fvl`   | +12.7 %| **−35.5 %** | no — tracks `cal_gint_rho` |

End-to-end (`Driver / atomic_world`): 5 of 6 cases net-positive
(+2.8 % to +8.9 %); only case 1 (`bxyz=27`) is still net-negative because of
the persistent NN regression below.

## In progress / next

### 2. NN template tuning for awkward `bxyz` values

**Problem:** `gemm_nn_vbatch` regresses by **−34 % at `bxyz=27`** and
**−29 % at `bxyz=80`**. Both are `bxyz` values that don't tile cleanly to
`BLK_N=64`:

- `bxyz=27`: 27 / 64 = 42 % useful in the single N-tile.
- `bxyz=80`: 80 = 64 + 16 → two N-tiles, the second 16/64 = 25 % useful.

The cases that *improve* (`bxyz=64`, `bxyz=125`) sit at clean tile
boundaries — strong evidence that the cliff is shape-fit, not algorithmic.

**Where to land the fix:** the shape-based dispatch refactor (item 1 above)
created the landing pad. Adding new NN tiers means **only** adding `else if`
arms to `gemm_nn_vbatch` in `dgemm_vbatch.cu` — no caller changes, no
header changes. Sketch:

```cpp
if (max_n <= 8) { /* tier 0 */ }
else if (max_m <= 32)  { /* NEW: BLK_N=32  for bxyz∈[27,32]  */ }
else if (max_m <= 64)  { /* tier 1: BLK_N=64                  */ }
else if (max_m <= 80)  { /* NEW: BLK_N=80  for bxyz∈[65,80]  */ }
else                   { /* tier 2: BLK_N=128 for bxyz>80     */ }
```

Open design questions:
- Is `BLK_N=80` viable, or does the kernel template require BLK_N to be a
  power of 2 / multiple of `DIM_Y`? Need to check
  `vbatched_gemm_nn_device` and the `THR_N = BLK_N / DIM_Y` derivation in
  `gemm_nn_vbatch.cuh`.
- For `bxyz=27`, BLK_N=32 is the natural fit, but a 32-wide N-tile may
  hurt occupancy on the 3090 vs the existing 64-wide tile. Worth trying
  BLK_N=32 with a wider BLK_M (e.g., 32) to keep the per-block thread
  count up.
- The middle tier (`max_m ≤ 64`) currently catches `bxyz ∈ {48, 64}` and
  bxyz=48 is also slightly slow (−8.8 %). Worth checking whether
  BLK_N=48 (if the template allows) helps it.

**Acceptance criteria:**
- All 6 cases ≥ baseline on `cal_gint_rho` and `cal_gint_fvl` (within ±2 %).
- No `cal_gint_vl` regression (the TN path is the known-good template, do
  not touch).
- Energies still match baseline to ≤ 1e-8 Ha on cases 1, 3, 6.

### 3. (Stretch) BLK_K tuning for `bxyz > 64` on TN

The new TN tier-2 template (`BLK 16x16x64`) lands within noise of the
existing `BLK_K=32` template at `bxyz=100, 125`. That suggests the K-axis
is not the bottleneck on those cases — likely shared-memory pressure from
the larger tile is offsetting the iter-count savings. Worth a brief
profile pass with `ncu` before declaring tier-2 final, but low priority
since current behavior is no-worse.

## Known to be out of scope (for now)

- **Re-evaluating the caller's 3-way DNF partition.** Now that the
  wrapper dispatches on shape, the caller could in principle partition
  more finely (or not at all) and let the wrapper handle everything.
  Defer until item 2 lands and we know what the right tier boundaries are.
- **`updateHk` apparent +13–17 % win.** This shows up in the new bench
  but is **not** explained by `cal_gint_vl` (which is +3–5 %). The gap is
  in non-Gint operators (`Overlap calculate_SR`, `EKinetic contributeHR`,
  `Nonlocal contributeHR`) — these have nothing to do with this branch.
  Treat as run-to-run GPU0 contention noise. See "Other timers" in
  `REPORT.md`.
