# P1 results: bench harness and weighted Ising format (ADR-015)

Prereg: `experiments/p1_bench/ADR-015.prereg-frozen.md`,
SHA-256 `d0f95bffd6ef3baef6c1b321984765cd60e6e5a460df91c5b4d3386765f8dd89`,
frozen 2026-09-28T05:52:12+02:00 at git HEAD 7308c62, before any harness code was written.
Every run below used the frozen parameters. Nothing was re-run with changed parameters, and no exploratory runs were done.

Machine: Xeon E5-2683 v4 (16C/32T), 23.9 GB RAM, Windows. Host-only; the GPU was not used.
Suite runs took place on 2026-09-28 from 05:59:44 to 06:01:17 (+02:00). The whole 33-cell suite took 90 s.

## Commands

```
build.bat                                   (adds [12/12] gset_bench)
check.bat                                   -> experiments\p1_bench\check_log.txt
build\gset_bench.exe --repro data\gset experiments\p1_bench\repro_G1-G5.csv   -> repro_log.txt
build\gset_bench.exe --suite experiments\p1_bench\instances.txt experiments\p1_bench 16 1800
                                            -> runs.csv, summary.csv, load.csv, suite_log.txt
build\gset_bench.exe data\gset\G<k>         (legacy mode, k=1..5) -> legacy_r1a.txt
```

## Verdicts (decisive rules of ADR-015 section 6)

| Rule | Verdict | Evidence |
|---|---|---|
| R1 bit-exact | **PASS** | (a) The legacy output of G1–G5 is byte-identical to the pre-change reference SHA-256 values (`legacy_r1a.txt`). (b) 40/40 assignments are identical to `anneal_branch`. Best cuts are 11613/11617/11621/11646/11616, and the best-assignment hashes match RESULTS-gset (`repro_log.txt`, `repro_G1-G5.csv`). |
| R2 consistency | **PASS** | cut == −E_tracked == −E_scratch in all 1,584 suite runs (0 inconsistent) and in all 40 repro runs. On 3 × K16 with weights in [−1000,1000], the integer energy matched brute force on all 2^16 states: 0 mismatches over 196,605 flips. |
| R3 coverage | **PASS** | All 11 instances loaded with matching SHA-256, n, m and weight counts (`load.csv`). 33/33 cells finished all R runs, and the slowest cell took 22.5 s against the 1800 s cap. |
| R4 TTS reported | **PASS** | Every cell has p, t, TTS99 and a bootstrap CI. CSV decimals are dots (`summary.csv`). |
| R5 green | **PASS** | `check.bat` exits 0 with 59 PASS lines and 0 FAIL: the 53 existing PASS lines (count unchanged) plus 6 new gset_bench self-check PASS lines. The run ends with "all green". |
| R6 prediction (targets reachable on sparse +1) | **FAIL** | G1 has p > 0 at both B10 (0.453) and B100 (0.938). **G22 has p = 0 at B100**: its best cut is 13358, one short of 13359, reached by 9 of 16 runs (5 runs at 13357; mean 13357.25). At B10, 4 of 64 runs reached 13358. The prediction was wrong for G22. |
| R7 prediction (K2000 unsolved, unscaled schedule) | **PASS** | K2000 has p = 0 at B1, B10 and B100. Best cuts are 32219, 32630 and 32906 against 33337. |

Result: 6 of 7 rules pass. R6 fails as registered.

Not decisive (reported only, per R2): on the K16 instances the engine's best of 8 at B10 reached only 10622/13777, 12434/12434 and 5050/6376 of the exact optimum. That is below 0.99 on 2 of 3. The frozen t_hot = 5 is tiny next to weights of order 1000, the same mismatch R7 predicted for K2000.

## Per-instance results

The gap is 100·(best − target)/target. A negative gap means the best cut is below the best-known value. No cut exceeded a target.

| Instance | n | m | Best-known | B1 best (gap %) | B10 best (gap %) | B100 best (gap %) |
|---|---|---|---|---|---|---|
| G1 | 800 | 19176 | 11624 | 11624 (0.000) | 11624 (0.000) | 11624 (0.000) |
| G2 | 800 | 19176 | 11620 | 11617 (−0.026) | 11620 (0.000) | 11620 (0.000) |
| G3 | 800 | 19176 | 11622 | 11621 (−0.009) | 11622 (0.000) | 11622 (0.000) |
| G4 | 800 | 19176 | 11646 | 11646 (0.000) | 11646 (0.000) | 11646 (0.000) |
| G5 | 800 | 19176 | 11631 | 11630 (−0.009) | 11631 (0.000) | 11631 (0.000) |
| G22 | 2000 | 19990 | 13359 | 13323 (−0.269) | 13358 (−0.007) | 13358 (−0.007) |
| G55 | 5000 | 12498 | 10299 | 10226 (−0.709) | 10279 (−0.194) | 10295 (−0.039) |
| G60 | 7000 | 17148 | 14188 | 14080 (−0.761) | 14156 (−0.226) | 14178 (−0.070) |
| G70 | 10000 | 9999 | 9595 | 9448 (−1.532) | 9546 (−0.511) | 9571 (−0.250) |
| G81 (±1) | 20000 | 40000 | 14060 | 13822 (−1.693) | 13900 (−1.138) | 13968 (−0.654) |
| K2000 (±1) | 2000 | 1999000 | 33337 | 32219 (−3.354) | 32630 (−2.121) | 32906 (−1.293) |

Best-known sources (ADR-015 section 3): G1–G5 come from Matsuda's SBM Gset benchmark, as in RESULTS-gset.md. G22 13359 comes from the BLS/Angers table. G55 10299 and G60 14188 come from the Angers MAMBP table and the 0816keisuke list. G70 9595 is from Zick, arXiv:2311.09275, and G81 14060 from Zick, arXiv:2505.18508. K2000 33337 is from Goto et al., Sci. Adv. 7, eabe7953 (2021).

## TTS99 where p > 0

TTS99 = t·ln(0.01)/ln(1−p). t is the mean per-run wall time, **measured with 16 runs sharing the CPU**, so it is not a single-run-alone time. The 95% CI is a bootstrap (B = 2000, splitmix64 seed 20260928). "inf" means the percentile landed on a resample with p* = 0.

| Instance | Budget | R | p | t (s) | TTS99 (s) | 95% CI (s) |
|---|---|---|---|---|---|---|
| G1 | B1 | 64 | 0.0156 | 0.0152 | 4.44 | [1.41, inf] |
| G1 | B10 | 64 | 0.4531 | 0.1234 | 0.94 | [0.65, 1.42] |
| G1 | B100 | 16 | 0.9375 | 1.1306 | 1.88 | [1.11, 3.12] |
| G2 | B10 | 64 | 0.0156 | 0.1217 | 35.59 | [11.38, inf] |
| G2 | B100 | 16 | 0.1875 | 1.1760 | 26.08 | [11.49, inf] |
| G3 | B10 | 64 | 0.1875 | 0.1158 | 2.57 | [1.57, 5.33] |
| G3 | B100 | 16 | 0.5625 | 1.1454 | 6.38 | [3.16, 14.03] |
| G4 | B1 | 64 | 0.0469 | 0.0142 | 1.36 | [0.58, inf] |
| G4 | B10 | 64 | 0.0469 | 0.1101 | 10.56 | [4.40, inf] |
| G4 | B100 | 16 | 0.0625 | 1.1693 | 83.43 | [25.68, inf] |
| G5 | B10 | 64 | 0.1406 | 0.1108 | 3.37 | [1.95, 7.96] |
| G5 | B100 | 16 | 0.3750 | 1.1808 | 11.57 | [5.53, 26.86] |

Every other cell has p = 0, so TTS99 is inf: G2/G3/G5 at B1, and all budgets of G22, G55, G60, G70, G81 and K2000.

## K2000 provenance

This is the real instance of Inagaki et al., Science 354, 603 (2016) (the 2000-spin all-to-all ±1 problem), not a synthetic stand-in.
- File `WK2000_1.rud`, fetched from https://raw.githubusercontent.com/hariby/SA-complete-graph/master/WK2000_1.rud
- SHA-256 `9ed615e5e18726914f12740b7f9bedb6b69676477ed5958fac42c8ab0c252ba7`, 22,775,141 bytes
- n = 2000 and m = 1,999,000, with 998,980 edges of +1 and 1,000,020 of −1 (Σw = −1040)
- The file is stored at `data/k2000/WK2000_1.rud` and is not committed (data/ is gitignored).

## Files

- `ADR-015.prereg-frozen.md`, `PREREG_SHA256.txt`: the frozen prereg and its hash
- `instances.txt`: the frozen instance list (path, target, SHA-256, n, m, weight counts)
- `load.csv`: per-instance load and hash check
- `runs.csv`: 1,584 rows, one per run (cut, E_tracked, E_scratch, consistent, success, wall_s, assignment SHA-256)
- `summary.csv`: 33 rows, one per cell
- `suite_log.txt`, `check_log.txt`, `repro_log.txt`, `repro_G1-G5.csv`, `legacy_r1a.txt`

## Refused / limits

- Legacy mode (`gset_bench <file>`, no flag) still refuses weights ≠ +1 by design, to keep R1(a) byte-exact. Signed weights go through `--suite`.
- The new loader refuses non-integer weights, self-loops, duplicate edges, out-of-range vertices and header/edge-count mismatches. The self-check exercises all 5 cases.
- No speed claim against other solvers and no schedule tuning (ADR-015 section 7). The failed R6 and the K16 shortfall are not "fixed" by re-running with other temperatures.
