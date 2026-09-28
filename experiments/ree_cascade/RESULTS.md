# ADR-013 results: rare-earth SX separation sequencing as a QUBO

Pre-registration: `ADR-013.prereg-frozen.md`, SHA-256
`b8c2d6575059e4952c320b110d729322c43e9ed200e72f675e04383b869510ed`, frozen 2026-09-28T04:53:25+02:00
(git HEAD 2747e94). Hash re-checked after the runs. Everything here is synthetic process-design
optimisation on published separation factors and a cited USGS ore composition. It says nothing about
any company's plant.

## Commands (PowerShell, repo root, PYTHONDONTWRITEBYTECODE=1)

```
nvcc -O3 -arch=sm_120 -std=c++17 -o build\check_ree_qubo.exe src\check_ree_qubo.cu   # via vcvars64
build\check_ree_qubo.exe experiments\ree_cascade 64            # frozen: 64 runs, 400 anneal + 20 hold
python tools\ree_neal.py experiments\ree_cascade 420 64         # neal 0.6.0, 420 sweeps, 64 reads, seed 12345
python tools\ree_analyze.py experiments\ree_cascade             # -> ree_summary.csv, ree_rules.json
python tools\render_ree.py experiments\ree_cascade              # -> figures\
# exploratory (not pre-registered): 10x longer schedules
build\check_ree_qubo.exe experiments\ree_cascade\exploratory_long_schedule 64 4000 200
python tools\ree_neal.py experiments\ree_cascade\exploratory_long_schedule 4200 64
python tools\ree_analyze.py experiments\ree_cascade\exploratory_long_schedule
```

Seeds: engine run seeds 1..64 per instance (best-of-8 = seeds 1..8); neal seed 12345, 64 reads
(best-of-8 = first 8 reads); random instances seed = 1000 n + k, k = 0..9, n = 4..12 (90 instances),
plus the 2 real instances (92 total). Wall times: engine ~28 s total, neal ~16 s total (frozen schedule).
The energy-check states and the brute-force enumeration are deterministic (see run_log.txt).

## Real instance (Mountain Pass GH2-57 feed, Haxel USGS OFR 2005-1219 Table 4)

Mole fractions La 0.300, Ce 0.537, Pr 0.039, Nd 0.114, Sm+ 0.010.

| instance | DP optimum (flow x stages) | greedy | greedy gap | engine best-of-8 | engine p_s | engine TTS99 | neal best-of-8 | neal p_s | neal TTS99 | R3 |
|---|---|---|---|---|---|---|---|---|---|---|
| MP_P507-A | 58.730 | 58.912 | 0.31 % | optimum | 0.172 | 11.7 ms | +10.2 % (missed) | 0.172 | 9.3 ms | engine win |
| MP_P507-B | 50.197 | 50.284 | 0.17 % | optimum | 0.172 | 9.5 ms | optimum | 0.141 | 8.7 ms | tie |

The optimal sequence is the same for both beta sets: La | Ce Pr Nd Sm+ (beta 8.0, 14 stages) -> Ce | Pr Nd Sm+
(46 stages for A, 36 for B) -> Pr Nd | Sm+ (12 stages) -> Pr | Nd (69 stages for A, 59 for B). Greedy
(the cheaper of the largest-beta-first and most-balanced heuristics) is only 0.2-0.3 % worse here.

## Frozen rules

| rule | verdict | numbers |
|---|---|---|
| R1 encoding exactness | **FAIL** (on the 1e-6 energy-agreement clause only) | Brute force passes: MP_P507-A (2^20 states) has 14 zero-penalty states, R5_0 (2^20) has 14, and R4_0 (2^10) has 5. These equal Catalan(n-1), every one is a valid tree, and the QUBO minimum equals the DP optimum on all three (58.7304 / 92.6170 / 93.6806). But engine `binary_energy` vs direct double-precision QUBO energy differs by up to 2.18e-5 (normalised units) and exceeds 1e-6 on 70/92 instances. Cause: `BinaryProblem.a` is float32, so each pair ray carries float(sqrt|Q|)^2 rounding. |
| R2 engine optimality | **FAIL** | Engine best-of-8 hits the optimum on both real instances, but on only 27.8 % of random instances (need >= 95 %). On 9 instances the best-of-8 is infeasible, so the mean gap over all instances is infinite; the mean gap over feasible best-of-8 is 12.7 % (need <= 1 %). neal for comparison: real A missed, real B hit, random 21.1 %, 3 infeasible. |
| R3 engine vs neal | **tie** | 36 wins, 12 ties, 44 losses; two-sided sign test p = 0.43 |
| R4 time-to-solution | **comparable** | Median TTS99(engine)/TTS99(neal) = 1.35 over the 32 instances where both have p_s > 0. Side tally: engine-only p_s > 0 on 11, neal-only on 7, neither on 42. |
| R5 vs greedy | **FAIL** | Greedy is suboptimal on 51/92 instances (mean gap 3.3 %, max 23.3 %). The engine best-of-8 reaches the optimum on only 9 of those 51 (neal: 8). |

My written prediction was R1 PASS, R2 PASS, R3 tie, R4 comparable. R3 and R4 came out as predicted;
R1 and R2 did not.

## Scaling with n (frozen schedule, 10 random instances per n; n=5 also includes the 2 real ones)

| n | vars | engine hit8 | neal hit8 | engine p_s | neal p_s | engine feasible | neal feasible | engine t/run | neal t/read | greedy suboptimal |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 10 | 10/10 | 10/10 | 0.281 | 0.264 | 0.77 | 0.76 | 186 us | 146 us | 2 |
| 5 | 20 | 9/12 | 7/12 | 0.116 | 0.130 | 0.86 | 0.86 | 400 us | 296 us | 7 |
| 6 | 35 | 6/10 | 2/10 | 0.045 | 0.038 | 0.90 | 0.91 | 788 us | 470 us | 4 |
| 7 | 56 | 1/10 | 1/10 | 0.009 | 0.013 | 0.94 | 0.94 | 1.38 ms | 754 us | 4 |
| 8 | 84 | 0/10 | 0/10 | 0.006 | 0.002 | 0.83 | 0.88 | 2.41 ms | 1.21 ms | 8 |
| 9 | 120 | 1/10 | 0/10 | 0.005 | 0.003 | 0.69 | 0.73 | 3.72 ms | 1.79 ms | 4 |
| 10 | 165 | 0/10 | 0/10 | 0 | 0 | 0.40 | 0.41 | 5.39 ms | 2.44 ms | 6 |
| 11 | 220 | 0/10 | 0/10 | 0 | 0 | 0.18 | 0.18 | 8.06 ms | 3.67 ms | 7 |
| 12 | 286 | 0/10 | 0/10 | 0 | 0 | 0.08 | 0.12 | 11.3 ms | 4.77 ms | 9 |

Median best-feasible gap (engine / neal): n=7 3.6 % / 3.7 %; n=8 12.9 % / 6.6 %; n=10 26.6 % / 17.4 %;
n=11 60 % / 32 %; n=12: no feasible engine best-of-8 on the median instance / 74 %.

## Exploratory (not pre-registered): 10x longer schedules

Engine 4000 + 200 sweeps, neal 4200 sweeps, same seeds. The picture is essentially unchanged.
R3 tie (37 / 10 / 45, p = 0.44). R4: median TTS ratio 2.08 over 40 instances, i.e. "engine slower"
under the frozen thresholds. Engine random hit8 23.3 % (neal 24.4 %). Best-of-8 infeasible on 1 instance
each. Engine reaches the optimum on 12 of the 51 greedy-suboptimal instances (neal 8). Longer annealing
does not fix it.

## Where it breaks / interpretation

- The penalty-dominated landscape is the problem. Each valid tree is an isolated zero-penalty point:
  going from one valid sequence to another needs several coordinated flips (remove a split, re-attach
  children), and every intermediate state pays >= P. Single-flip annealing (the engine and neal alike)
  freezes into whichever tree it reaches first. The feasible fraction collapses from ~0.9 at n=6-7 to
  ~0.1 at n=12 because the constraint chain gets longer.
- The exact DP is O(n^3) and solves every instance here exactly with no search (at n = 12 only 66 multi-component sub-mixtures and 286 (sub-mixture, split) pairs; its runtime was not separately timed). For this
  problem a QUBO/annealing formulation is strictly worse than the classical exact method. The most
  useful output is the ranking of greedy vs optimum (greedy is 0-23 % off, 3.3 % on average).
- Engine vs neal: statistically indistinguishable in quality. The engine is ~2x slower per sweep (it
  carries the ray machinery for a problem with only pair couplings), and TTS is within the "comparable"
  band on the frozen schedule.
- R1's float32 energy mismatch (<= 2.2e-5 normalised) does not change any hit or gap: costs and hits are
  recomputed in double from the returned bit string. It is still a FAIL under the rule as written.

## Caveats

Sharp adjacent splits only, constant betas, a Fenske-type stage count (not a Kremser extraction+scrub
design), no Ce(IV) pre-separation, cost = flow x stages proxy. The Ce/La beta of 8.0 comes from a weak
source (thesis). Pr in the feed is interpolated in the source table. TTS is single-thread wall time on
this PC.

## Figures (experiments/ree_cascade/figures/)

fig_ree_flowsheet.png (optimal sequence for both beta sets), fig_ree_landscape.png (all 2^20 energies of
MP_P507-A, with the 14 feasible trees), fig_ree_convergence.png (engine energy-vs-sweep traces),
fig_ree_solvers.png (p_s, gap, time and TTS vs n), fig_ree_gap_scatter.png (engine vs neal gaps).
