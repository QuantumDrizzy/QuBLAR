# Known limits (living list)

Every entry is a measured or code-verified limit, with the file or number that shows it. Entries are never
deleted: when a limit is fixed, its status changes and the fix is referenced. Times are Madrid (UTC+2).

Status values: **OPEN** (limit holds), **MITIGATED** (partly addressed, residual stated), **FIXED** (closed,
with evidence), **SUPERSEDED** (replaced by another path), **POLICY** (a hardware/operational limit handled by a
rule, not by code).

| ID | Short name | Status |
|---|---|---|
| [KL-001](#kl-001) | GPU full-load thermal/acoustic limit (RTX 5060 Ti at 180 W cap) | POLICY / OPEN |
| [KL-002](#kl-002) | Ising engine was CPU-only before P2 | MITIGATED |
| [KL-003](#kl-003) | 32-bit indices cap sparse problems at ~2^31 directed entries | OPEN (P4) |
| [KL-004](#kl-004) | Fixed t_hot = 5 freezes annealing on large weights | MITIGATED |
| [KL-005](#kl-005) | float32 couplings broke ADR-013 R1 energy agreement | FIXED for integer instances / OPEN for real-valued |
| [KL-006](#kl-006) | Legacy `gset_bench` mode: one shared coupling, refuses ±1 | SUPERSEDED |
| [KL-007](#kl-007) | LYTH cannot express QuBLAR's sparse kernels | OPEN |
| [KL-008](#kl-008) | Unibit QPU/QGPU/QRAM are docs only | OPEN |
| [KL-009](#kl-009) | No CI, no releases, proprietary licence | OPEN |
| [KL-010](#kl-010) | Windows-only build with hard-coded paths | OPEN |
| [KL-011](#kl-011) | P2 (ADR-016) interrupted: G81/K2000 never ran, no verdict | OPEN |

---

<a id="kl-001"></a>
## KL-001 — GPU full-load thermal/acoustic limit

- **Date:** 2026-09-28
- **Symptom:** during the P2 bench (`gpu_anneal --bench`), the RTX 5060 Ti ran at 100 % utilisation at its
  180 W power limit with about 20 % of VRAM in use, and the machine got very loud. The owner stopped the run at
  **06:16 (UTC+2)**.
- **Evidence:**
  - `nvidia-smi --query-gpu=power.limit,power.max_limit,power.default_limit` (read-only, idle, 2026-09-28):
    180.00 W / 180.00 W / 180.00 W. The card already runs at its maximum board power; there is no headroom to
    raise and no lower default.
  - `experiments/p2_gpu/bench_log.txt` stops after the G81 load line; the P2 CSVs were last written 06:15:10.
  - Utilisation, VRAM share and noise are the owner's observation. **No power/temperature telemetry was logged
    during P2** — that gap is itself part of this entry.
- **Impact:** heavy GPU runs (16,384 replicas, 100 % occupancy, minutes long) are not acceptable while the owner
  is at the machine. P2 is incomplete because of it (KL-011). Any throughput number measured at 100 % load may
  also be clock/thermal-dependent, and we have no record of clocks.
- **Workaround / policy:**
  1. Heavy GPU runs need the owner's explicit OK, or are scheduled when the owner is away.
  2. Otherwise they must be capped: fewer replicas per launch, a duty cycle (sleep between launches), or launch
     pacing — declared in advance as an execution note, never as a changed experimental parameter.
  3. Every GPU bench logs `nvidia-smi` power, temperature, SM clock and utilisation (e.g. 1 s sampling to a CSV
     next to the results) from now on.
  4. Agents do not start GPU runs on their own; `check.bat` also counts, because it now runs GPU self-checks.
- **Status:** POLICY in force; OPEN until a paced/logged bench mode exists.

<a id="kl-002"></a>
## KL-002 — The Ising engine was CPU-only before P2

- **Date:** found 2026-09-28 (scale audit, 05:36–06:00)
- **Symptom:** the Ising/QUBO path had no GPU code; the 16 GB GPU was idle during inference.
- **Evidence:** `src/ising_recon.hpp:202-225` `anneal_branch()` is sequential single-spin Metropolis on the host
  (double precision); `branch_fractions()` (`:229-247`) parallelises only across branches with `std::thread`,
  capped at 16 (`check_ising.cu:108`). No `__global__` in the Ising path (27 kernels exist, all in ray, detector,
  transient, muon and scenario code). Audit numbers: 3.49e6 spin updates/s on one thread (G1); 16 threads saturate
  at about 1–2e9 edge visits/s.
- **Impact:** throughput ceiling of the host engine; no TTS competitive with hardware solvers.
- **Workaround:** P2 added `src/ising_gpu.cuh` + `tools/gpu_anneal.cu` for the P1 integer format (not for the
  imaging `BinaryProblem`). Measured on G1/G22: 1.114e10 and 9.53e9 spin updates/s, 28.77× and 18.66× the
  16-thread CPU rate, bit-identical results (`experiments/p2_gpu/RESULTS_PARTIAL.md`).
- **Status:** MITIGATED. Residual: the imaging engine (`anneal_branch`, `BinaryProblem`) is still CPU-only; the
  GPU engine is unverified on G81 (global state) and K2000 (dense) and has no ADR verdict (KL-011).

<a id="kl-003"></a>
## KL-003 — 32-bit indices cap sparse problems at ~2^31 directed entries

- **Date:** 2026-09-28 (audit; P4 in the roadmap)
- **Symptom:** a symmetric CSR stores each undirected edge twice; with 32-bit indices the structure cannot exceed
  2^31 directed entries, i.e. **~1.07e9 undirected edges, regardless of VRAM**.
- **Evidence:**
  - `src/ising_recon.hpp:45,48`: `BinaryProblem::row_ptr` and `nbr_ptr` are `std::vector<int>` — a hard 2^31
    nonzero cap for the imaging engine.
  - The P1/P2 weighted path is better but not proven: `WeightedGraph::row` is int64 (`src/ising_weighted.hpp:39`),
    and the device kernels use int64 row offsets (`src/ising_gpu.cuh:154,235`), but columns, vertices and weights
    are int32, the loader caps n ≤ 2^30, per-spin fields must fit int32, and nothing larger than K2000
    (4.0e6 directed entries) has ever been run. Until P4 tests 64-bit offsets end to end, 1.07e9 edges is the
    planning cap.
  - `docs/benchmarks/vram_planning.csv`: at 10 B per undirected edge, 0.9 × 16 GiB already holds 1.55e9 edges, so
    the index cap, not VRAM, binds on every card from 16 GB up (calculated, not measured).
- **Impact:** buying more VRAM does not raise the sparse problem size until P4 lands.
- **Workaround:** none needed at current sizes (largest instance: G81, 4e4 edges; K2000, 2e6 edges).
- **Status:** OPEN — planned in P4 (64-bit offsets, multi-spin coding, measured peak VRAM vs model).

<a id="kl-004"></a>
## KL-004 — Fixed t_hot = 5 freezes annealing on large weights

- **Date:** 2026-09-28 (P1 finding)
- **Symptom:** the frozen Gset/MAP schedule (t_hot = 5, t_cold = 1e-3) is far too cold when local fields are
  large; the anneal starts frozen and gets stuck.
- **Evidence (`experiments/p1_bench/RESULTS.md`, `summary.csv`):** K2000 (|f| ~ 45) p = 0 at all three budgets,
  best 32219 / 32630 / 32906 vs 33337 (−3.35 / −2.12 / −1.29 %), as predicted by ADR-015 R7. On K16 instances with
  weights in [−1000, 1000], best of 8 at B10 reached only 10622/13777 and 5050/6376 of the exact optimum.
- **Impact:** quality on weighted or dense problems is limited by the schedule, not the algorithm.
- **Workaround:** P2 (ADR-016 §4) uses weight-scaled temperatures, T_hot = σ = sqrt((2/n)·Σw²), T_cold =
  0.1·w_min, identically on CPU and GPU (K2000: T_hot = 44.71).
- **Status:** MITIGATED in code. Evidence so far is only on +1 graphs (G22 reached 13359 on both devices, P1 never
  did). **Not yet shown on large weights**: K2000 never ran in P2 (KL-011), so ADR-016 R7 is undecided.

<a id="kl-005"></a>
## KL-005 — float32 couplings broke ADR-013 R1 energy agreement

- **Date:** 2026-09-28 (ADR-013 run)
- **Symptom:** the pair-ray encoding stores general couplings as float32 ray amplitudes (`BinaryProblem.a`), so the
  engine's energy differs from the exact energy.
- **Evidence:** `docs/adr/ADR-013-ree-separation-qubo.md` §9: engine `binary_energy` vs direct double energy differs
  by up to 2.18e-5 (normalised), > 1e-6 on 70/92 instances → R1 FAIL.
- **Impact:** energies from the imaging engine are not exact on general QUBOs (hits and gaps in ADR-013 were
  recomputed in double, so its results are unaffected).
- **Workaround / fix:** P1 (ADR-015) added an exact integer path (`src/ising_weighted.hpp`: int32 weights, int64
  energy). ADR-015 R2: 0 inconsistent of 1,584 suite runs; exact on all 2^16 states of 3 K16 instances.
- **Status:** FIXED for integer-weighted instances. OPEN for real-valued QUBOs such as ADR-013's (their
  coefficients are not integers; they still go through the float32 `BinaryProblem`). Roadmap I1 (`qubo/0.1`,
  expansion in double) is the planned fix.

<a id="kl-006"></a>
## KL-006 — Legacy `gset_bench` mode: one shared coupling, refuses ±1

- **Date:** 2026-09-28 (audit)
- **Symptom:** `gset_bench <graph>` maps MaxCut onto `BinaryProblem` with a single uniform coupling
  `lambda = -1` and exits 2 on any weight other than +1 ("edge weight -1 is not +1").
- **Evidence:** `tools/gset_bench.cu` (legacy path); audit: G81, K2000 and every ±1 Gset graph refused.
- **Impact:** about half of Gset and K2000 could not be run.
- **Workaround:** `gset_bench --suite <list> <outdir> ...` (P1) runs any integer-weighted graph through the
  weighted engine; G81 and K2000 ran in P1 (`experiments/p1_bench/summary.csv`). The legacy mode is kept on
  purpose, byte-identical, for ADR-015 R1(a).
- **Status:** SUPERSEDED by `--suite`.

<a id="kl-007"></a>
## KL-007 — LYTH cannot express QuBLAR's sparse kernels

- **Date:** 2026-09-28 (audit)
- **Symptom:** LYTH has no integer buffers, no gather/indexed loads, no `select`, no atomics.
- **Evidence:** LYTH ADR-0026 and language docs; QuBLAR's own comments decline LYTH for the accept step
  (`src/ising_recon.hpp:183-185, 216`; `src/check_local_field.cu:7`).
- **Impact:** a CSR sweep and the Metropolis accept cannot be written in LYTH; QuBLAR → LYTH integration has no code.
- **Workaround:** only dense contractions are expressible; roadmap I2 needs one branch-free `select`/`sign` op.
- **Status:** OPEN.

<a id="kl-008"></a>
## KL-008 — Unibit QPU/QGPU/QRAM are documentation only

- **Date:** 2026-09-28 (audit)
- **Symptom:** the Unibit (MTLB) repo has a 256-bit ISA emulator with a cost model (the CPU rung); QPU, QGPU and
  QRAM exist only as ADR text (ADR-0002 "Proposed").
- **Evidence:** no state-vector simulator or QPU code in `Unibit/src`; `ising_energy.uasm` runs on all-zero data at
  n = 256 and reads nothing from QuBLAR.
- **Impact:** any "QuBLAR → QGPU/QPU" statement is a plan, not a result.
- **Status:** OPEN.

<a id="kl-009"></a>
## KL-009 — No CI, no releases, proprietary licence

- **Date:** 2026-09-28 (audit)
- **Evidence:** no `.github/`, no tags or releases; `LICENSE`: proprietary, all rights reserved (the GitHub repo
  is public). No unit-test framework; invariants are PASS/FAIL lines inside the check programs, and `check.bat`
  runs only a subset of the programs.
- **Impact:** nobody else can install or independently verify; regressions are caught only when someone runs
  `check.bat` by hand.
- **Status:** OPEN (roadmap R1–R4; the licence is the owner's decision).

<a id="kl-010"></a>
## KL-010 — Windows-only build with hard-coded paths

- **Date:** 2026-09-28 (audit)
- **Evidence:** `build.bat` hard-codes the VS 2022 BuildTools and OptiX SDK 9.1.0 paths; `tools/branches_blaze.py`
  imports Blaze from an absolute `C:\Users\...` path. No CMake, pip, conda or Docker. Several check programs and
  (before P1) `gset_bench` were not in `build.bat`.
- **Impact:** the build works on this one machine.
- **Status:** OPEN (roadmap R1: CMake, OptiX optional, 0 absolute paths).

<a id="kl-011"></a>
## KL-011 — P2 (ADR-016) interrupted: G81/K2000 never ran, no verdict

- **Date:** 2026-09-28, 06:16 (UTC+2)
- **Symptom:** the P2 bench was stopped by the owner (KL-001) after G1 and G22; G81 had loaded and started,
  K2000 never started.
- **Evidence:** `experiments/p2_gpu/bench_log.txt` (last line: G81 load), `summary.csv` / `throughput.csv` (G1, G22
  only), `docs/adr/ADR-016-p2-gpu-anneal.md` (no section 9). Frozen prereg unchanged (SHA-256 `a63419f6…c6e16`).
- **Impact:** none of R1–R8 can be decided: each is stated over all four instances, and R5 (`check.bat` green with
  the P2 lines, P1 intact) was never run. The measured half is in `experiments/p2_gpu/RESULTS_PARTIAL.md`.
- **Workaround:** resume only G81 and K2000 with the frozen parameters, under the KL-001 policy, with telemetry;
  then run R5 and fill ADR-016 §9. G1/G22 are not re-run.
- **Status:** OPEN. Committed as WIP/INTERRUPTED.
