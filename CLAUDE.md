# Project: abacus-gemm-opt

Working repo for optimizing the batched-GEMM kernels in ABACUS's Gint module
(`gemm_nn_vbatch` / `gemm_tn_vbatch` in
`source/source_lcao/module_gint/kernel/dgemm_vbatch.cu`).

The code under optimization is invoked from `phi_operator_gpu.cu`:
- `phi_mul_phi` (Hamiltonian build, V_loc) → **`gemm_tn_vbatch`**
- `phi_mul_dm`  (charge density from DM)   → **`gemm_nn_vbatch`**

`M` of the vbatch GEMMs scales with `mgrids_num_ ≈ bxyz = bx*by*bz`.
`N`/`K` scale with the per-element orbital count `nw`. So a meaningful benchmark
must vary **both** `bxyz` *and* the species mix.

---

## Builds

| name      | binary                                                   | purpose                |
|-----------|----------------------------------------------------------|------------------------|
| baseline  | `/home/dzc/abacus/abacus-gemm-opt/build-baseline/abacus_2g` | reference (do not modify) |
| test      | `/home/dzc/abacus/abacus-gemm-opt/build-test/abacus_2g`     | the build under optimization |

After editing `dgemm_vbatch.cu`, rebuild **only** the test build:

```bash
cmake --build /home/dzc/abacus/abacus-gemm-opt/build-test -j
```

---

## Benchmark suite

Location: `~/abacus/test-examples/ABACUS-test/LCAO_test/gemm_vbatch_bench/`

Six 108-atom cases derived from the `008_Li27Ni9O54Mn9Co9` reference structure.
Same lattice / k-point / cutoff in every case — only `bx,by,bz` and the species
mix change, so cross-case deltas isolate to the vbatch GEMM shapes.

| # | Case dir                  | bx,by,bz | bxyz | ntype | Composition (108 atoms)                  |
|---|---------------------------|---------|------|-------|------------------------------------------|
| 1 | `case1_bxyz27_5species`   | 3,3,3   |  27  | 5     | Li27 Ni9 O54 Mn9 Co9                     |
| 2 | `case2_bxyz48_6species`   | 4,4,3   |  48  | 6     | Li27 Ni9 O54 Mn6 Fe3 Co9                 |
| 3 | `case3_bxyz64_6species`   | 4,4,4   |  64  | 6     | Li27 Ni6 Al3 O54 Mn9 Co9                 |
| 4 | `case4_bxyz80_7species`   | 5,4,4   |  80  | 7     | Li27 Ni6 Fe3 O54 Mn6 Al3 Co9             |
| 5 | `case5_bxyz100_7species`  | 5,5,4   | 100  | 7     | Li24 Na3 Ni6 Cu3 O54 Mn9 Co9             |
| 6 | `case6_bxyz125_8species`  | 5,5,5   | 125  | 8     | Li24 Na3 Ni6 Cu3 O54 Mn6 Fe3 Co9         |

All cases run with `scf_nmax=2`, `cal_force=1`, `cal_stress=1`, `nspin=2`,
`gint_precision=double`, `ks_solver=genelpa`, Γ-only.
This exercises `cal_gint_vl` four times, `cal_gint_rho` twice, `cal_gint_fvl`
once per run.

### Running the suite

The runner is **serial** (one case after another, baseline before test):

```bash
cd ~/abacus/test-examples/ABACUS-test/LCAO_test/gemm_vbatch_bench
./run_bench.sh
```

The launch line, identical for every run, is:

```
OMP_NUM_THREADS=7 mpirun -n 1 --bind-to socket <abacus_2g>
```

Each invocation produces, inside the case dir:

```
run_baseline.log    run_test.log         # stdout
wall_baseline.txt   wall_test.txt        # wall time captured by the harness
OUT.baseline/       OUT.test/            # ABACUS output dirs (renamed from OUT.ABACUS)
   └── running_scf.log                   # source of all kernel timings
```

### Where to look in the timing table

Inside each `running_scf.log`, the `TIME STATISTICS` section near the end has
the rows that matter:

| timer                       | what it covers                                | which kernel |
|-----------------------------|-----------------------------------------------|--------------|
| `Gint cal_gint_vl`          | `phi_mul_phi` during H build                  | `gemm_tn_vbatch` |
| `Gint cal_gint_rho`         | `phi_mul_dm` during `dm2rho`                  | `gemm_nn_vbatch` |
| `Gint cal_gint_fvl`         | force/stress contribution from V_loc          | both flavors |
| `HamiltLCAO updateHk`       | superset of `cal_gint_vl`                     | (sanity)     |
| `HSolverLCAO solve`         | SCF inner loop (includes `cal_gint_rho`)      | (sanity)     |
| `Driver atomic_world`       | end-to-end                                    | (sanity)     |

`updateHk`, `elpa_solve`, `cal_force_stress` should not move under this
optimization — if they do by more than ±2%, suspect noise or a build mismatch
before declaring victory.

### Generating the report

`run_bench.sh` only runs the binaries; it does not parse. To produce a Markdown
report from the existing `OUT.{baseline,test}/running_scf.log` files, the
parsing recipe is:

1. Read the `TIME STATISTICS` block from each log.
2. For each case, pull out the 8 timers in the table above for both builds.
3. Compute `speedup = baseline / test`.
4. Tabulate per-kernel and end-to-end.

A reference report is at:

```
~/abacus/test-examples/ABACUS-test/LCAO_test/gemm_vbatch_bench/REPORT.md
```

and a machine-readable parse dump at:

```
~/abacus/test-examples/ABACUS-test/LCAO_test/gemm_vbatch_bench/_parsed.json
```

---

## Known baseline characterization (as of 2026-04-09 sweep)

The "test" build at the time the suite was created had this profile vs
baseline (single-shot, RTX 3090, GPU0):

- **`gemm_tn_vbatch` (`cal_gint_vl`):** uniformly **+3.4 % to +5.5 %** across all
  6 cases. Treat this kernel as the *known-good* template.
- **`gemm_nn_vbatch` (`cal_gint_rho` / `cal_gint_fvl`):** highly shape-sensitive.
  Best case +3.5 %, **worst case −27 % on `cal_gint_rho` and −24 % on `cal_gint_fvl`**.
  The worst regressions hit `bxyz=27` and `bxyz=80`. The two cases that
  *improved* were `bxyz=64` (4³) and `bxyz=125` (5³) — both perfect cubes.
  Strong hint: the optimized `gemm_nn_vbatch` makes a tile/block choice that
  assumes `M = bxyz` is a "nice" multiple of the kernel tile and falls off a
  cliff for awkward sizes.

This characterization will become stale as you optimize. Re-run
`run_bench.sh` and refresh `REPORT.md` after any non-trivial change to
`dgemm_vbatch.cu`.

---

## Workflow checklist for each optimization iteration

1. Edit `source/source_lcao/module_gint/kernel/dgemm_vbatch.cu`.
2. `cmake --build build-test -j` (do **not** rebuild `build-baseline`).
3. `cd ~/abacus/test-examples/ABACUS-test/LCAO_test/gemm_vbatch_bench && ./run_bench.sh`
4. Re-parse the 12 `running_scf.log` files and update `REPORT.md`.
5. **Verify correctness** before claiming a speedup: diff the SCF energies in
   `OUT.baseline/running_scf.log` vs `OUT.test/running_scf.log` for at least
   the cases that improved — a kernel that "speeds up" by skipping work will
   show up as an energy mismatch here.

---

## Notes / gotchas

- The runner deletes `OUT.ABACUS` and renames the result to `OUT.baseline` /
  `OUT.test`. Don't put precious artifacts in `OUT.ABACUS` between runs.
- All `.upf` / `.orb` files in each case dir are **symlinks** into
  `~/abacus/test-examples/PP_ORB`. If that directory moves, the cases break.
- GPU 0 is shared across the system. If a number jitters, re-run with
  `CUDA_VISIBLE_DEVICES=0` or pin the run to a known-idle device.
- `scf_nmax=2` is intentional — the goal is to measure kernels, not converge
  the SCF. Don't bump it without a reason; it changes the call counts of every
  Gint timer in the table above.
