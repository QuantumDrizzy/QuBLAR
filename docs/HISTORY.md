# History

Dated log of what was done, measured and decided. Newest last. Times are Madrid (UTC+2) and come from git commit
dates, `PREREG_SHA256.txt` freeze stamps, results files and file modification times. Verdicts are quoted as
recorded in each ADR; see [KNOWN_LIMITS](KNOWN_LIMITS.md) for open problems.

Earlier phases (LiDAR, NLOS, external data, muon tomography, the Ising engine) are documented in
`docs/RESULTS-phase*.md` and ADR-001..009; this log starts on 2026-09-27.

## 2026-09-27

| time | event | record |
|---|---|---|
| 04:23 | README: the name QμBLAR and its logo | `e2bb5a7` |
| 21:09 | Ising: mine/ahead checks, shared `local_field_dE`, the walk into the measured void | `2747e94` |

## 2026-09-28

### ADR-010..014: five pre-registered experiments (pushed)

Each prereg was frozen (SHA-256 recorded) before any of its code ran; section 9 of each ADR was filled after.
All five were committed between 05:32 and 05:35 and pushed; `master` = `origin/master` = **`7308c62`**.

| ADR | frozen | verdict (as recorded) | commit | results |
|---|---|---|---|---|
| [ADR-010](adr/ADR-010-ark-falsification-test.md) ark muography falsification | 03:51 | hull hypotheses B1–B3 distinguishable, **3/3**; min. detectable void 2 m (marginal), 3 m robust | `3a0a523` | [RESULTS](../experiments/ark/RESULTS.md) |
| [ADR-011](adr/ADR-011-khufu-muography-validation.md) Khufu muography validation | 04:12 | detection, localisation, blind **PASS (3/4 criteria)**; paper comparison partly consistent **18/25** | `b2bfa33` | [RESULTS](../experiments/khufu/RESULTS.md) |
| [ADR-012](adr/ADR-012-tunnel-nlos-ghost.md) tunnel NLOS + ghost imaging | 04:38 | **1/7 PASS** (headline as committed; §9 lists A4 PASS and the B1-pred prediction check PASS) | `4a90be8` | [RESULTS](../experiments/tunnel/RESULTS.md) |
| [ADR-013](adr/ADR-013-ree-separation-qubo.md) REE separation sequencing as QUBO | 04:53 | **0/3 pass/fail rules PASS** (R1, R2, R5 FAIL); R3 tie and R4 comparable, as predicted | `72c1232` | [RESULTS](../experiments/ree_cascade/RESULTS.md) |
| [ADR-014](adr/ADR-014-ree-carbonatite-muography.md) REE carbonatite muography | 05:08 | **3/6 PASS** (R2, R4, R6; R1, R3, R5 FAIL) | `9d7dafe` | [RESULTS](../experiments/ree_muography/RESULTS.md) |

Also: `9bf26e7` (05:32) ignores raw experiment volumes and keeps frozen preregs byte-exact (`-text`); `7308c62`
(05:35) indexes ADR-009..014 in the README.

### Scale audit (05:36–06:00, read-only)

Audit of QuBLAR, LYTH and Unibit for scaling (report kept outside the repo; no tracked file changed). Findings that
shaped the next steps: the Ising engine is CPU-only sequential Metropolis (1–2e9 edge visits/s on 16 threads);
there are no per-edge couplings, so `gset_bench` refuses ±1 graphs (G81, K2000, half of Gset); no TTS harness;
32-bit indices; no CI, no library, Windows-only build. It proposed the roadmap P1 → P2 → P3 → I1..I4 → R1..R4.
Entries KL-002, KL-003, KL-006..KL-010 come from it.

### P1 / ADR-015: bench harness and weighted integer Ising format

| time | event |
|---|---|
| 05:52:12 | prereg frozen (`experiments/p1_bench/ADR-015.prereg-frozen.md`, SHA-256 `d0f95bff…dd89`), before any harness code |
| 05:59 | `check.bat`: 59 PASS (53 existing + 6 new), 0 FAIL; repro of G1–G5 |
| 05:59:44–06:01:17 | 33-cell suite (11 instances × B1/B10/B100), 1,584 runs, 90 s, CPU only |
| 06:02 | results written |

**Verdict: 6 of 7 rules PASS.** R1 bit-exact reproduction PASS; R2 exact consistency PASS (0 of 1,584
inconsistent); R3 coverage PASS (11/11 instances incl. G81 and K2000, 33/33 cells); R4 TTS reported PASS; R5 green
PASS; **R6 FAIL** (G22 best 13358 vs 13359 at B100, p = 0); R7 PASS (K2000 unsolved by the unscaled schedule, best
32906). Details: [experiments/p1_bench/RESULTS.md](../experiments/p1_bench/RESULTS.md),
[ADR-015](adr/ADR-015-p1-bench-harness.md). Findings: KL-004 (t_hot = 5 too cold), KL-005 fixed for integer
instances, KL-006 superseded.

### P2 / ADR-016: GPU annealing engine — partial, INTERRUPTED

| time | event |
|---|---|
| 06:07:22 | prereg frozen (`experiments/p2_gpu/ADR-016.prereg-frozen.md`, SHA-256 `a63419f6…c6e16`), before any kernel code |
| 06:09–06:12 | `src/ising_gpu.cuh`, `tools/gpu_anneal.cu` written |
| ~06:15 | G1 and G22 measured (GPU batch, then matched-wall CPU pool); CSVs written 06:15:10; G81 loaded and started |
| **06:16** | **owner stopped the run**: RTX 5060 Ti at 100 % utilisation at its 180 W cap, very loud (KL-001) |

Measured before the stop: G1 1.114e10 vs 3.872e8 spin updates/s (**28.77×**), G22 9.53e9 vs 5.107e8
(**18.66×**); CPU/GPU assignments identical 584/584 and 892/892; oracle 0/2000 mismatches; G22 target 13359
reached on both devices with the weight-scaled schedule. **G81 unfinished, K2000 never ran, ADR-016 has no
verdict.** Details: [experiments/p2_gpu/RESULTS_PARTIAL.md](../experiments/p2_gpu/RESULTS_PARTIAL.md). KL-011.

### Documentation (after the stop, no GPU use)

`docs/KNOWN_LIMITS.md`, this file, `experiments/p2_gpu/RESULTS_PARTIAL.md`, and `docs/benchmarks/` (charts
rendered on the CPU from the committed CSVs by `tools/render_benchmarks.py`, plus a VRAM planning model marked as
calculated estimates). P1 and the P2 partial were committed locally; nothing pushed.
