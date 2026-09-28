# P2 partial results: GPU annealing engine (ADR-016) — INTERRUPTED, NO VERDICT

> **Status: INTERRUPTED. ADR-016 has NO verdict.** The owner stopped the run on 2026-09-28 at 06:16 (UTC+2)
> because the RTX 5060 Ti sat at 100 % utilisation at its 180 W power cap and was very loud
> (see [KL-001](../../docs/KNOWN_LIMITS.md#kl-001)). G1 and G22 finished; **G81 had loaded and started
> and did not finish; K2000 never ran.** None of the eight rules R1–R8 can be decided, because every
> rule is stated over all four instances (or needs `check.bat`, which was not re-run for P2).
> This file reports the measured half as data, not as a verdict. Nothing here was re-run.

- Prereg: `experiments/p2_gpu/ADR-016.prereg-frozen.md`, SHA-256
  `a63419f608fecdbc8f201da2c7bb2cca91d0b72e55d319a7f1752a193c9e6c16` (frozen 06:07:22 +02:00 at HEAD 7308c62,
  before any P2 kernel code). Unchanged; verified again before committing. `docs/adr/ADR-016-p2-gpu-anneal.md`
  has no section 9.
- Machine: RTX 5060 Ti 16 GB (sm_120, 36 SM, 180 W cap), Xeon E5-2683 v4 16C/32T, Windows, CUDA 13.0.
- Raw data (all written 06:15:10 +02:00): `summary.csv`, `throughput.csv`, `agreement.csv`, `load.csv`,
  `runs_gpu.csv` (32,768 rows), `runs_cpu.csv` (1,476 rows), `bench_log.txt` (ends at the G81 load line).
- Not recorded: GPU power, temperature or clocks during the run. From now on every GPU bench must log them
  (KL-001 policy).

## Setup (as frozen in ADR-016 §4)

Same algorithm on both devices: the P1 integer Metropolis engine, sweep order 0..n−1, splitmix64 draw stream,
weight-scaled temperatures T_hot = σ, T_cold = 0.1·w_min, 4000 anneal + 200 hold sweeps. GPU: one warp per
replica, 16,384 replicas (seeds 1..16384) in one batch. CPU: the unmodified P1 engine on 16 threads, seeds
started in order until the GPU batch wall time is used up (matched wall).

## Measured (G1 and G22 only)

| | G1 GPU | G1 CPU | G22 GPU | G22 CPU |
|---|---:|---:|---:|---:|
| n / m | 800 / 19,176 | | 2,000 / 19,990 | |
| plan | CSR, w int8, f int8, shared state | host | CSR, w int8, f int8, shared state | host |
| T_hot / T_cold | 6.924 / 0.100 | same | 4.471 / 0.100 | same |
| runs | 16,384 | 584 | 16,384 | 892 |
| batch wall (s) | 4.941 | 5.067 | 14.441 | 14.673 |
| spin updates / s | **1.114e10** | 3.872e8 | **9.530e9** | 5.107e8 |
| speedup (rate GPU / CPU) | **28.77×** | | **18.66×** | |
| best cut (target) | 11624 (11624) | 11624 | **13359 (13359)** | **13359** |
| gap to best-known | 0.000 % | 0.000 % | 0.000 % | 0.000 % |
| mean cut | 11617.29 | 11617.94 | 13349.86 | 13349.75 |
| successes / runs (p) | 10,461 / 16,384 (0.6385) | 380 / 584 (0.6507) | 65 / 16,384 (0.00397) | 3 / 892 (0.00336) |
| amortised t = wall / runs (s) | 0.000302 | 0.008677 | 0.000881 | 0.016449 |
| **amortised TTS99 (s), 95 % CI** | **0.00137 [0.00134, 0.00139]** | 0.0380 [0.0341, 0.0421] | **1.02 [0.81, 1.33]** | 22.5 [9.6, inf] |
| latency (s) (GPU: batch wall; CPU: mean per-run wall) | 4.941 | 0.137 | 14.441 | 0.262 |
| latency TTS99 (s), not decisive | 22.4 | 0.60 | 16,730 | 358 |
| cut == −E_tracked == −E_scratch | 16,384 / 16,384 | 584 / 584 | 16,384 / 16,384 | 892 / 892 |

Kernel time: G1 4,937.1 ms in 56 launches of 76 sweeps; G22 14,433.3 ms in 140 launches of 30 sweeps.

### CPU/GPU agreement and energy oracle (`agreement.csv`)

| instance | shared seeds | identical assignments (SHA-256) | oracle states | oracle mismatches |
|---|---:|---:|---:|---:|
| G1 | 584 | **584** | 1000 | **0** |
| G22 | 892 | **892** | 1000 | **0** |

The GPU engine is bit-identical to the CPU engine on every shared seed, and the independent device energy
kernel agrees with the host on all 2000 random states.

## How to read the two TTS numbers

- **Amortised TTS99** (the ADR-016 definition) treats the batch as a stream of attempts: t = batch wall / runs.
  It is a *throughput* measure: how much wall time the machine spends per 99 %-confidence success when many
  runs execute concurrently. It favours the GPU because 16,384 runs share one 14 s batch.
- **Latency TTS99** uses the time until *one* run's answer is available (the whole batch for the GPU, since no
  replica finishes early; the mean per-run wall for the CPU). With the GPU's p = 0.004 on G22 this formula gives
  16,730 s, which is pessimistic: it ignores that one batch holds 16,384 attempts.
- The practical wall-clock to a G22 target cut on this GPU is **one batch, 14.4 s** (65 of 16,384 replicas hit
  13359; the chance that a batch of 16,384 at p = 0.004 contains none is about e^−65). On the CPU, 3 of 892 runs
  hit it in the same 14.7 s.
- **No claim against Toshiba's SBM or any other solver.** Goto et al. (Sci. Adv. 7, eabe7953, 2021) report a
  99 % TTS of 2.7 s for dSBM (FPGA) on G22. Our 1.02 s amortised figure is not the same quantity (different
  definition of t, different hardware, no statement about single-run latency), and our minimum wall to any
  G22 answer is 14.4 s. ADR-016 §7 excludes solver comparisons; this file does too.

## What these numbers do and do not say

- They say: on G1 and G22 the GPU runs the identical algorithm 18.7–28.8× faster than 16 CPU threads, stays
  exact, and reaches the G22 target that P1 never reached (P1 used the unscaled t_hot = 5, t_cold = 1e-3; here
  both devices reach 13359 with the scaled temperatures, so the gain on G22 quality is the schedule, not the GPU).
- They do not say: anything about G81 (global-state path, n = 20,000) or K2000 (dense path, ±1, 2e6 edges);
  whether R3 (≥ 10× on *all four*) holds; whether R7 (K2000 ≥ 33170) holds; whether `check.bat` is still green
  with the P2 lines (R5 was never run).
- Rates would be lower for problems that do not fit shared memory; 1.11e10/s is a small-graph, shared-state
  figure.

## Resuming P2 (needs the owner's OK — KL-001)

Re-run only the unfinished instances with the frozen parameters, logging `nvidia-smi` power/temperature/clocks,
at a time the owner approves (or when away), or with a pacing/duty-cycle cap declared in advance as an
execution note (not a parameter change). G1/G22 are not re-run. Then fill ADR-016 §9 with all eight verdicts.
