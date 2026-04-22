# A complete tour of `vbatched_gemm_nn_impl` / `_tn_impl` parameters

> **Status note (A100 retune, 2026-04-19):** the production kernels are now
> `gemm_{nn,tn}_vbatch_pipelined.cuh`, not the originals covered by most of
> this document. They use `cp.async` (SM80+) with 2-stage shared-memory
> double buffering and wider `BLK_K` tiles targeted at A100. The
> originals are kept in the tree as a rollback target for one release
> cycle. The tuning rationale below is preserved as *historical
> context* — most of the 3090-specific conclusions (Section 3c) no
> longer drive the dispatch. The A100-specific updates live in the new
> section **7. A100 retune (2026-04-19)** at the bottom of this doc;
> Section 5 notes the expected 3090 regression.

Everything below refers to the (now-legacy) kernel template at `source/source_lcao/module_gint/kernel/gemm_nn_vbatch.cuh` (NN) and its sibling `gemm_tn_vbatch.cuh` (TN). They are classic Volkov-style "block × thread × register" SGEMM kernels (this layout originated in MAGMA / Volkov's paper). The dispatcher in `dgemm_vbatch.cu` picks one template instantiation per call.

I'll first walk through what each of the 9 template parameters *physically* means, then derive the resource cost equations, then show how to choose them for our shapes, and finally discuss porting to A100 / H100.

---

## 1. The 9 template parameters

The instantiations look like:

```
vbatched_gemm_nn_impl<T,  DIM_X, DIM_Y,  BLK_M, BLK_N, BLK_K,  DIM_XA, DIM_YA,  DIM_XB, DIM_YB>
```

There are **two thread layouts** that share the same physical block:

- **Compute layout** `DIM_X × DIM_Y` — used in the inner FMA loop.
- **Load layout for A** `DIM_XA × DIM_YA` — used to copy `A` from global → shared.
- **Load layout for B** `DIM_XB × DIM_YB` — used to copy `B` from global → shared.

A hard invariant: `DIM_X*DIM_Y == DIM_XA*DIM_YA == DIM_XB*DIM_YB` (every thread participates in every phase, just under a different 2-D index). Look at `gemm_nn_vbatch.cuh:46-55`:

```
int idt  = DIM_X*idy + idx;     // linear thread id
int idxA = idt % DIM_XA;        // re-shape into the A-loader
int idyA = idt / DIM_XA;
int idxB = idt % DIM_XB;        // re-shape into the B-loader
int idyB = idt / DIM_XB;
```

The same threads pretend to be a `DIM_X × DIM_Y` grid during compute, a `DIM_XA × DIM_YA` grid while loading A, and a `DIM_XB × DIM_YB` grid while loading B.

### 1a. `BLK_M`, `BLK_N`, `BLK_K` — the tile

These are the dimensions of the **C tile** (`BLK_M × BLK_N`) that one thread block produces, and the **K-strip width** (`BLK_K`) it consumes per main-loop iteration. Concretely:

| object | shape | lives in |
|---|---|---|
| C tile | `BLK_M × BLK_N` | registers `rC[THR_N][THR_M]` |
| A panel | `BLK_M × BLK_K` | shared memory `sA` |
| B panel | `BLK_K × BLK_N` | shared memory `sB` |

The block sweeps the K dimension in chunks of `BLK_K`. Inside each chunk it does an outer-product accumulation `rC += rA ⊗ rB`. The grid is therefore `(ceil(M/BLK_M), ceil(N/BLK_N), batchCount)` — see `gemm_nn_vbatch.cuh:400-402`.

### 1b. `DIM_X`, `DIM_Y` — compute thread layout

The block has `DIM_X * DIM_Y` threads. Each thread owns a sub-tile of C of size

```
THR_M = BLK_M / DIM_X    (rows per thread)
THR_N = BLK_N / DIM_Y    (cols per thread)
```

This is the **register tile**. Each thread keeps `THR_M*THR_N` accumulators in `rC[THR_N][THR_M]` (`gemm_nn_vbatch.cuh:61`). The whole inner-product loop boils down to `THR_M*THR_N` FMAs per K-step per thread, fed by `THR_M + THR_N` register loads (the rA / rB vectors at lines 151–161). This is what makes the kernel *compute-bound*: by raising `THR_M*THR_N` you get more FMAs per shared-memory load.

It must divide evenly: `BLK_M % DIM_X == 0` and `BLK_N % DIM_Y == 0`. If not, you get wasted threads and (worse) the unrolled loops over `THR_M`/`THR_N` no longer cover the whole tile.

### 1c. `DIM_XA`, `DIM_YA`, `DIM_XB`, `DIM_YB` — load thread layouts

A and B have to be staged into shared memory. The `(DIM_XA, DIM_YA)` shape is just *how the same `DIM_X*DIM_Y` threads are arranged when copying* the `BLK_M × BLK_K` panel of A (similarly for B).

Two constraints:

1. The product must equal the block size: `DIM_XA*DIM_YA = DIM_X*DIM_Y`.
2. Each panel must be tileable by the load layout in *whole strides*:
   - For A: `BLK_M % DIM_XA == 0` and `BLK_K % DIM_YA == 0`
   - For B: `BLK_K % DIM_XB == 0` and `BLK_N % DIM_YB == 0`

Why a separate loader layout? Because the natural compute layout `DIM_X × DIM_Y` (e.g. `8×16`) is rarely the best layout for a *coalesced* read of A. A is column-major in this kernel, so consecutive threads should walk consecutive rows of A to coalesce — that means you want the `x`-stride of the loader to match the leading dimension of A. By picking `DIM_XA = 16` you get warps of 32 threads scanning two contiguous rows of A at a time.

The loader you see in our config (`16, 8, 16` for nn) means: "load A as a 16-wide × 8-tall thread mesh" → coalesces along the M direction; "load B as 8-wide × 16-tall" → coalesces along the K direction (B is also column-major in the kernel because the impl swaps A↔B before launching, see comment at `gemm_nn_vbatch.cuh:387-389`).

### 1d. Derived parameters (the kernel computes them)

```
THR_M = BLK_M / DIM_X       // C-rows per thread
THR_N = BLK_N / DIM_Y       // C-cols per thread

slda  = BLK_M + 1           // shared lda, +1 to break shared-memory bank conflicts
sldb  = BLK_K + 1
shared_mem = (BLK_M+1)*BLK_K + (BLK_K+1)*BLK_N    elements of T
```

The `+1` padding (`gemm_nn_vbatch.cuh:289-290`, `:392-393`) is the standard SMEM bank-conflict avoidance trick: stride 32 (or any multiple of 32) is catastrophic on Ampere because every column hits the same 32 banks.

---

## 2. Hardware cost of each parameter

Let me make the cost model explicit. Per kernel block:

| resource | formula | comment |
|---|---|---|
| Threads / block | `DIM_X * DIM_Y` | must be ≤ 1024 |
| Registers / thread | ≈ `THR_M*THR_N + THR_M + THR_N + ra+rb load buffer + ~10 scratch` | the rC accumulators dominate |
| Shared memory / block | `((BLK_M+1)*BLK_K + (BLK_K+1)*BLK_N) * sizeof(T)` | 8 B for double, 4 B for float |
| Global loads / K-step | `BLK_M*BLK_K + BLK_K*BLK_N` | shared by all threads in the block |
| Shared loads / K-step | `BLK_K * (THR_M + THR_N)` | per thread |
| FMAs / K-step | `BLK_K * THR_M * THR_N` | per thread |
| Arithmetic intensity | `(BLK_M*BLK_N*BLK_K) / ((BLK_M+BLK_N)*BLK_K * sizeof(T))` | classic GEMM AI |

Two derived ratios drive everything:

- **Compute intensity per shared read** = `THR_M * THR_N / (THR_M + THR_N)`. This must be ≥ 4–8 to keep the FMA pipes busy on Ampere. For our `DIM_X=8, DIM_Y=16, BLK_M=16, BLK_N=64` case, `THR_M=2, THR_N=4` → ratio = 8/6 ≈ 1.33. *That's low.* It's the price we pay for a tiny `BLK_M=16` (to fit the small nw axis). On a "real" GEMM you'd see `THR_M=THR_N=8`, ratio ≈ 4.
- **Bytes loaded from DRAM per output element** = `(M*K + K*N)/(M*N) * sizeof(T) = sizeof(T)*(1/N + 1/M)*K`. This is what kills you on small M or small N: as either dimension shrinks, you pay K-related DRAM traffic against fewer output elements. Our shapes are *exactly* the bad regime.

### 2a. The four hard limits (they decide occupancy)

On a 3090 SM (compute capability 8.6), per SM:

- **Max threads:** 1536 (48 warps)
- **Max blocks:** 16
- **Registers:** 65536 (32-bit)
- **Shared memory:** 100 KB total, but kernel-usable defaults to 48 KB unless you opt into the larger carveout
- **Max regs/thread (without spills):** ≈ 255

For a block to *actually* run, all four of these must fit. Occupancy = `min` of all four caps divided by total resources. Each cap that hits forces concurrent-block count down, which exposes more warp latency and hurts performance.

Worked example for the production NN tier 1 instantiation `<8,16, 16,64,16, 8,16, 8,16>` with `T=double`:

```
threads/block = 8*16          = 128                      → cap allows 1536/128 = 12 blocks
shared/block  = (16+1)*16 + (16+1)*64                    elements
              = 272 + 1088    = 1360 doubles = 10.6 KB   → cap allows 48/10.6 ≈ 4 blocks
regs/thread   ≈ 8 (rC) + 2 + 4 + load buffers + scratch ≈ 40 → cap allows 65536/(128*40) ≈ 12 blocks
blocks limit  = 16                                         → cap allows 16
```

The binding constraint is **shared memory: 4 blocks/SM**. That's 4 × 128 = 512 threads = 16 warps active per SM, or **33% occupancy** — on the low end for a kernel that leans this hard on shared memory loads.

Now consider tier 2 (`BLK_N=128`): shared memory becomes `272 + (16+1)*128 = 272 + 2176 = 2448` doubles = 19.1 KB → only 2 blocks/SM. Occupancy halves. *But* the block does ~2× the work, so total throughput is the same on paper — and you save on K-loop overhead because each block now reuses A more times. That's the trade tier 2 is making at `bxyz ≥ 80`.

### 2b. The K-loop cost

Each block executes `K / BLK_K` iterations of the main loop. Each iteration does:

- One `__syncthreads()` pair
- `BLK_M*BLK_K + BLK_K*BLK_N` global loads (split across the block)
- `BLK_K * THR_M * THR_N` FMAs per thread

Doubling `BLK_K` halves the iteration count and halves the per-iteration sync overhead. That's why TN tier 2 (`BLK_K=64` instead of `32`) helps for `bxyz ≥ 80`: the K dimension *is* `bxyz` in the TN mapping (see comment at `dgemm_vbatch.cu:97-99`), so the K loop trip count grows linearly with bxyz, and the per-iter sync becomes a real cost.

You cannot make `BLK_K` arbitrarily large though: shared memory grows linearly in `BLK_K`, and the inner FMA loop is `#pragma unroll` over `BLK_K` (`gemm_nn_vbatch.cuh:147`). Beyond ~64, the unrolled loop blows out the instruction cache and registers spill.

---

## 3. How to *choose* parameters for a given shape

I'll walk through the actual decision the dispatcher makes for our cases, then generalize.

### 3a. NN (`gemm_nn_vbatch`): `M = bxyz`, `N = nw2`, `K = nw1`

After the swap inside `_impl` (line 387), the kernel sees:

```
kernel-M = nw2 (small,  2..27)
kernel-N = bxyz (large, 27..125)
kernel-K = nw1 (small,  2..27)
```

So `BLK_M` should be sized against the nw2 distribution, `BLK_N` against bxyz, and `BLK_K` against nw1 — but **not** by the naive "≥ max" rule. The actual trade is more subtle:

#### Why "BLK_M ≥ max nw2" is the *wrong* rule

Look at the inner FMA loop at `gemm_nn_vbatch.cuh:147-172`:

```c
for (k = 0; k < BLK_K; k++) {
    for (m = 0; m < THR_M; m++) rA[m] = sA(m*DIM_X + idx, k);
    for (n = 0; n < THR_N; n++) rB[n] = sB(k, n*DIM_Y + idy);
    for (n = 0; n < THR_N; n++)
        for (m = 0; m < THR_M; m++)
            rC[n][m] += rA[m] * rB[n];
}
```

**Every thread computes its full `THR_M × THR_N` register tile, unconditionally.** The boundary check only happens at the C-store on line 244. So when `BLK_M > nw2`, the dead rows of C are still:

- loaded into `sA` (the clamped `fetch` macro makes it correct, but bandwidth is spent),
- multiplied in the inner loop (FMAs are spent),
- carried in `rC` registers (register pressure is spent).

Only the *write-back* is suppressed.

That gives us two regimes with fundamentally different cost profiles:

| | BLK_M too large (`BLK_M ≥ max nw2`) | BLK_M too small (`BLK_M < max nw2`) |
|---|---|---|
| compute waste | `(BLK_M − nw2)/BLK_M` of every FMA is dead, on every matrix | only the boundary tile has waste |
| A traffic | dead rows of A still loaded | none |
| B traffic | each matrix's B is loaded once | each matrix's B is loaded `⌈nw2/BLK_M⌉` times (redundant across M-tiles) |
| shared memory | grows with BLK_M → lower occupancy | smaller per-block → higher occupancy |
| register tile | `THR_M*THR_N` grows → register pressure | smaller |
| block launches per matrix | 1 in M | `⌈nw2/BLK_M⌉` in M |

So the real trade is **wasted compute + wasted A loads + lower occupancy** (big tile) vs. **redundant B loads + more block launches** (small tile).

#### Worked dead-row table for our nw2 distribution

nw2 is wide and skewed across species — roughly `{9, 13, 17, 27}` depending on which atoms are in a sub-batch. Compute the dead-row fraction for each candidate `BLK_M`:

| nw2 | BLK_M=16 dead frac | BLK_M=32 dead frac |
|---|---|---|
| 9  | 7/16 = 44% | 23/32 = 72% |
| 13 | 3/16 = 19% | 19/32 = 59% |
| 17 | 15/32 ≈ 47% (1st tile full, 2nd tile fills 1/16) | 15/32 = 47% |
| 27 | 5/32 = 16%  (1st tile full, 2nd tile fills 11/16) | 5/32 = 16% |

The row that matters: **at nw2=27, BLK_M=16 and BLK_M=32 have *the same* total dead-row fraction.** The wasted area is the same — split into "one half-empty boundary tile" (BLK_M=16) vs. "one tile that's partially-empty everywhere" (BLK_M=32). For every *smaller* nw2, BLK_M=16 wastes strictly less. So on dead-row waste alone, BLK_M=16 dominates BLK_M=32.

The only thing BLK_M=32 has going for it is "no redundant B loads at nw2=27". B is `BLK_K × BLK_N = 16 × 64 = 1024 doubles ≈ 8 KB`, so the redundant load is one extra ~8 KB transfer per nw2>16 matrix — small. Meanwhile BLK_M=32 also doubles `sA` shared memory and doubles the per-thread register tile, both of which *cut occupancy in half* on this kernel.

#### Why this resolves the way it does on 3090 (and would flip on A100)

On **3090** with its 1/64 fp64 rate, dead FMAs are *the* most expensive resource we spend — the kernel is fp64-throughput-bound, not bandwidth-bound. Burning 50% of the FMA pipe on dead rows is far worse than reloading 8 KB of B once. So BLK_M=16 wins decisively.

On **A100/H100** the fp64 pipe is 16–32× wider, so dead-FMA waste is much cheaper while bandwidth per FLOP is comparatively worse. The calculus flips: redundant B loads start to matter more than dead FMAs, and a slightly larger `BLK_M` (or a `BLK_M` chosen to exactly tile a *common* nw2 such as 13 or 17, not the maximum) becomes attractive. This is one of the things to retune when porting.

#### The corrected rule

Pick `BLK_M` to **minimize the *average* dead-row fraction across the actual nw2 distribution of the workload**, weighted by how often each nw2 occurs and biased by how expensive a dead FMA is on the target hardware. Heuristics:

- *Compute-bound device (3090 fp64)*: pick the smallest `BLK_M` that doesn't drop the warp-occupancy floor. Boundary blocks are cheap; dead FMAs are not.
- *Bandwidth-bound device (A100/H100 fp64)*: pick `BLK_M ≈ median(nw2)` so the typical matrix is one tile. Boundary reloads of B start to matter; dead FMAs are cheap.

The same logic applies to `BLK_K` (driven by the nw1 distribution) and `BLK_N` (driven by the bxyz distribution — simpler because bxyz is fixed per run, so you only choose between "one big tile" and "two smaller tiles" without any distribution mixing).

Look at the three tiers in `dgemm_vbatch.cu:67-83`:

| tier | trigger | BLK_M (nw2) | BLK_N (bxyz) | BLK_K (nw1) | covers |
|---|---|---|---|---|---|
| 0 | `max_n ≤ 8` | 8 | 64 | 16 | sub-batches with only Li/B/C/N/O/F |
| 1 | `max_n > 8 && max_m ≤ 64` | 16 | 64 | 16 | bxyz ∈ {27, 48, 64} with full nw range |
| 2 | `max_n > 8 && max_m > 64` | 16 | 128 | 16 | bxyz ∈ {80, 100, 125} |

**Why `BLK_M=16` and not 32?** Because nw2 ≤ 27 in our datasets. A `BLK_M=32` tile would compute 32 rows of C but only 27 of them are valid → 16% of the threads are useless and we still waste shared-memory traffic on the dead rows. Two `BLK_M=16` tiles waste at most ~31% on the second tile (when nw2=17), and the *first* tile is 100% useful — so total useful work = (16 + min(11, BLK_M))/32 ≈ 84% which is better than 27/32 ≈ 84% but with the bonus of running on fewer threads → smaller blocks → higher occupancy.

**Why `BLK_N=64` for bxyz ≤ 64 and `BLK_N=128` for bxyz > 64?** A single N-tile is always better than two (no extra A-panel reload, no boundary waste). Doubling BLK_N doubles shared memory used by `sB` and the register tile in N → it cuts occupancy in half, but at bxyz ≥ 80 the alternative is two tiles and the second tile is up to 50% wasted (bxyz=80 → second tile only fills 16/64 = 25%; bxyz=125 → 61/64 = 95%). At bxyz=125 it's a clear win; at bxyz=100 it's a wash; at bxyz=80 it's still a small win because the K-loop overhead amortizes.

**Why `BLK_K=16`?** nw1 ≤ 27. Same logic as `BLK_M`: anything bigger forces a partial K-tile. `BLK_K=16` means *all* matrices with `nw1 ≤ 16` skip the main loop entirely and only run the tail branch (line 205+) — that's a big win for the small-Z species (Li, B, C, N, O, F) which are the majority of atoms in the test cases.

### 3b. TN (`gemm_tn_vbatch`): `M = nw1`, `N = nw2`, `K = bxyz`

Now the K dimension is the *large* one (bxyz), and the M, N dims are *small*. So:

- `BLK_M`, `BLK_N` should match nw, not bxyz → keep them at 16
- `BLK_K` should match bxyz → 32 for small bxyz, 64 for large bxyz

This is exactly the dispatch in `dgemm_vbatch.cu:126-142`. Note the asymmetric-pair safety note at line 113-124: even for large bxyz we *don't* widen `BLK_M`/`BLK_N`, because the dominant pair shapes are 13×27 (TM-O) and 7×27 (TM-Li), and a 32×32 tile would idle 60% of threads on those pairs. The win from doubling `BLK_K` (halving the K-loop trip count) is unconditional regardless of pair shape, so that's the *only* knob the new tier turns.

### 3c. The general recipe for a new shape

Given a target `(M, N, K)`:

1. **Pick the tile so each axis is hit by 1–2 tiles**, preferring 1 if you can afford the shared memory.
2. **Pick `BLK_M`/`BLK_N` so `THR_M*THR_N ≥ 8`** (the more, the better, until registers spill). On 3090 doubles, target `THR_M*THR_N` in [8, 32].
3. **Pick `DIM_X`, `DIM_Y` so each is a multiple of 8 (warp packing)** and `DIM_X*DIM_Y` is in {64, 128, 256}. 128 is the sweet spot for occupancy on shared-memory-bound kernels.
4. **Pick the loader shapes so `DIM_XA` matches the leading dimension of A in cache lines** (i.e. `DIM_XA * sizeof(T) ≥ 32 B = one sector`) and likewise for B.
5. **Compute resources, compare to the four caps (threads, blocks, regs, smem), and check that occupancy is ≥ 25%.**
6. **If smem is the binding constraint and you can't grow occupancy, grow the register tile instead** — moving work from shared to registers buys you AI for free.

Step 6 is exactly why "more registers per thread" is often the right move on this kernel: we are already shared-memory-occupancy-bound, so spending more registers doesn't cost us anything until we hit the 255-reg-per-thread spill cliff.

---

## 4. How the hardware caps shift between 3090 / A100 / H100

| resource | RTX 3090 (SM 8.6) | A100 (SM 8.0) | H100 (SM 9.0) |
|---|---|---|---|
| FP64 throughput / SM | **2 FMA/cycle** (1/64 of FP32) | **32 FMA/cycle** (full rate) | 64 FMA/cycle |
| Threads/SM | 1536 | 2048 | 2048 |
| Blocks/SM | 16 | 32 | 32 |
| Regs/SM | 64K | 64K | 64K |
| SMEM/SM | 100 KB (48 default) | 164 KB (up to 163) | 228 KB |
| L1/SM | 128 KB | 192 KB | 256 KB |

The single most important fact: **3090 has *catastrophically* bad fp64**. Its fp64 rate is 1/64 of fp32 — so on 3090 a *double-precision* GEMM is compute-bound at very modest arithmetic intensity, and there's a real ceiling no amount of tile tuning can break. On A100/H100, fp64 is full-speed (or full half-speed on H100 for HMMA), so the kernel turns *back* into a memory-bandwidth-bound problem and you need bigger tiles to feed the wider FP64 pipes.

Practical implications when porting our tile choices:

- **3090 → A100**: SMEM cap nearly doubles, so you can afford `BLK_N=128` even at tier 1. The fp64 pipe is 16× wider, so the register tile should grow: `THR_M*THR_N` should rise from ~8 to ~32. That means doubling `BLK_M` and/or `BLK_N` while keeping `DIM_X*DIM_Y` constant. For the NN kernel, going from `<8,16, 16,64,16,...>` to `<8,16, 32,128,16,...>` would be a sensible first step (4× the per-thread work, same threads, same launch shape), but only if nw2 ≥ 32 — otherwise the wider M-tile is just waste.
- **A100 → H100**: H100 wants either tensor cores or much bigger tiles still. The async-copy path (`cp.async`) exists on A100 too but is *required* on H100 to overlap loads with compute. Our kernel does *not* use `cp.async`; on A100/H100 a 1.3–1.7× win is sitting on the table from rewriting the global→shared loads as `cp.async` two-stage pipelines. That's a structural change, not a parameter tweak.
- **The asymmetric-pair issue gets worse on bigger GPUs.** Their occupancy budget is bigger so widening BLK_M/BLK_N becomes feasible *in principle*, but our smallest pair (7×27) doesn't get any larger, so the "wasted thread" fraction stays the same. The TN dispatcher's choice to keep `BLK_M=BLK_N=16` would still be correct on A100 for our shapes; only `BLK_K` would grow further (probably to 128).

Rule of thumb: **on a new GPU, do not start by re-tuning tiles. First check which of {threads, smem, regs, blocks} is your binding constraint on the new device, then move the tile in the direction that relaxes that constraint while preserving `THR_M*THR_N`.**

---

## 5. The two specific things this kernel is leaving on the table

While we're here, two things you should know about the *current* design:

1. **Atomic store in the writeback** (`gemm_nn_vbatch.cuh:248`): `atomicAdd(C + offsC, rC[n][m] * alpha)`. This is required because multiple sub-batches may write the same C tile (the dispatch from `phi_operator_gpu.cu` produces sub-batches that share C buffers). Atomic doubles on global memory go through a slow path on 3090 (not on A100 — A100 has fast fp64 atomics). It's not free anywhere, but on 3090 it's a noticeable tax. If we could colour the batches so different sub-batches never collide on C, we could use a plain store and probably pick up a few percent on tier 1.

2. **No double-buffered shared memory.** Look at the main loop at line 115. It does load → sync → compute → sync → load, with one shared buffer. A double-buffered version (load *next* while computing *this*) hides the global-load latency with compute and is the standard MAGMA recipe — but it doubles the smem footprint, which on 3090 doubles is what we're already bound on. So we'd need to *either* shrink BLK_K (hurts) *or* opt into the 100 KB smem carveout via `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, 100*1024)`. Worth experimenting with on A100 where it's basically free.

---

## 6. Known regression: 3090 under the A100-targeted kernels

As of 2026-04-19, the production dispatch (`dgemm_vbatch.cu`) launches the new `gemm_{nn,tn}_vbatch_pipelined.cuh` kernels. These are tuned for A100's wider FP64 pipe and larger SMEM budget, not for 3090.

Running the retuned kernel on 3090 is expected to regress against the previous 3090-tuned dispatch, for architectural reasons:
- 3090 runs FP64 at 1/64 of FP32. The kernel remains compute-bound, so the "wider tiles + cp.async overlap" wins that pay off on A100 can't help here — the compute pipe saturates before any load latency becomes visible.
- The 2-stage shared-memory buffering doubles SMEM per block (~45–80 KB), pushing occupancy from the old ~4 blocks/SM to ~1–2 blocks/SM on 3090. The lower occupancy exposes warp latency that the old kernel did not.
- `cp.async` itself is slower on 3090 (it's present on sm_86 but the memory pipe is narrower than A100's).

Expected magnitude, projected from the dead-FMA cost model above:
- `cal_gint_rho` (NN): regression of 15–40%
- `cal_gint_vl` (TN): regression of 5–15%
- `cal_force_stress` / `elpa_solve`: ±2% (unchanged — these do not call the vbatch kernels)

A regression >60% is not architectural mismatch — it likely indicates a bug in the pipelined kernel. Sanity-check in that case.

This regression is **accepted** as the cost of targeting A100. Reverting to the 3090-tuned dispatch means editing `dgemm_vbatch.cu` to launch `vbatched_gemm_{nn,tn}_impl` (the unpipelined versions) instead of the `_pipelined_impl` variants; both kernel bodies are still in the tree.

---

## 7. A100 retune (2026-04-19)

### 7a. What changed

- **Kernel body:** a cp.async-pipelined rewrite lives in
  `gemm_nn_vbatch_pipelined.cuh` and `gemm_tn_vbatch_pipelined.cuh`.
  Global→shared copies now use `__pipeline_memcpy_async` with
  `__pipeline_commit` / `__pipeline_wait_prior(1)` to maintain one
  in-flight group while compute consumes the current stage.
- **Shared memory:** two stages of `sA`/`sB`, allocated back-to-back in
  the dynamic SMEM block. Size roughly doubles; `cudaFuncSetAttribute`
  with `cudaFuncAttributeMaxDynamicSharedMemorySize` opts the kernels
  into the larger per-block carveout (up to 163 KB on A100,
  up to 99 KB on 3090) when the request exceeds the 48 KB static cap.
- **Tile sizes (NN):** `BLK_K` raised 16 → 32 across the dispatch
  ladder. This halves the K-loop trip count and doubles tile
  arithmetic intensity on the nw1 axis. The nw2 axis (`BLK_M`) and
  bxyz axis (`BLK_N`) ladders are preserved — the same shape-bucket
  rationale from Section 3a still applies in caller-space.
- **Tile sizes (TN):** `BLK_K` raised 32 → 64 for small-bxyz, 64 → 128
  for large-bxyz. Halves the K-loop trip count on the bxyz axis, which
  is where this kernel spends most of its time. The asymmetric-pair
  branches (Tier Am / Tier An) are preserved — widening the small nw
  axis still wastes threads on TM-O and TM-Li shapes on any GPU.
- **Minimum compute capability:** sm_80 (was sm_60). Pre-Ampere GPUs
  lack `cp.async` and would fail to compile the pipelined kernel.
  Enforced in top-level `CMakeLists.txt`.

### 7b. Why this particular parameter choice

The A100 datasheet drives the design:
- **164 KB SMEM / SM, 108 SMs:** wider tiles become affordable.
  `(BLK_M+1)·BLK_K + (BLK_K+1)·BLK_N` doubles because of the 2 stages,
  but 2 blocks/SM still fit for every bucket in the dispatch, which
  keeps SM-level parallelism reasonable.
- **FP64 at full rate (~9.7 TFLOP/s):** dead-FMA waste is no longer
  dominant. Section 3c's "BLK_M=16 because dead FMAs dominate" argument
  inverts on A100; wider tiles are net-positive. Even so, we kept
  BLK_M (nw2 axis) at 8 or 16 because nw2 ≤ 27 — widening to 32
  has the same dead-row fraction as two 16-tiles on nw2=27, but costs
  more register pressure. The win from widening BLK_M on A100 is
  small relative to the win from widening BLK_K, which is why BLK_K
  is the main knob that moved.
- **HBM2e at 1555 GB/s:** the AI required for compute-bound is ~3-4
  FLOP/byte on A100 for FP64. With our shapes, we're around 2 FLOP/byte
  even at the widened tiles, so the kernel is bandwidth-bound. This is
  the regime where `cp.async` overlap is the single largest lever.

### 7c. Validation status

The tile sizes in `dgemm_vbatch.cu` are chosen by **architectural reasoning**, not by an empirical sweep on A100. A100 hardware was not available at retune time. Deferred to whoever runs on A100 first:
- Refine the (BLK_M, BLK_N, BLK_K, DIM_*, stages) grid per shape bucket
- Check whether 3-stage pipelining beats 2-stage for the TN buckets
  with BLK_K=128 (where two chunks in-flight might hide more latency)
- Decide whether BLK_M=32 or DIM_X×DIM_Y = 16×16 = 256 threads is
  worth the register-pressure hit

3090 correctness testing is the only validation performed in this tree.

### 7d. Rollback

If the A100 retune turns out to be broken or regresses more than expected:
1. Edit `dgemm_vbatch.cu` — replace every `vbatched_gemm_*_pipelined_impl` call with the corresponding `vbatched_gemm_*_impl` call.
2. Revert the `BLK_K` values in those calls to the original ladder (NN: 16 everywhere; TN: 32 for k≤64, 64 for k>64).
3. Restore the includes `#include "gemm_nn_vbatch.cuh"` / `"gemm_tn_vbatch.cuh"` at the top.
No CMake change is strictly required for rollback — the non-pipelined kernels are portable to sm_60+, but keeping sm_80+ as the minimum is fine until someone needs to run on older hardware.
