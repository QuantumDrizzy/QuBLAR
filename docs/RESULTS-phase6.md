# Phase 6 results: binary branches (ADR-007)

Measured on the RTX 5060 Ti host (32 threads, the engine using 16), with
`build\check_ising.exe`. The pass-or-fail checks are A to C and G; D to F report
measurements, and the ROI check is in the export step. All pass.

## The oracle

A 16-variable problem is enumerated exhaustively (2¹⁶ configurations). The annealer cooled
to MAP reaches the exact ground state, E = 7.404 in both. The region-of-interest QUBO
matches the full energy for 8 random configurations to 1.4 × 10⁻¹⁰.

## The void, against exposure

The problem has 306,328 binary variables and 75,594 rays, with 3.95 M ray-voxel entries.
The declared priors are λ = 2, κ = ln(999) = 6.91 (p₀ = 10⁻³), and there are 16 branches.

| muons / chamber | MLEM (ADR-006) | found "does not exist" (correct) | centroid to the void's axis | IoU | false voids, empty pyramid | evidence: data vs prior (nats) |
|---|---|---|---|---|---|---|
| 2²⁵ | 30 m off | 1 (0) | 62 m | 0 | 1 | 303 < 421 |
| 2²⁶ | 30 m off | 3 (3) | 0.3 m | 0.09 | 0 | 466 > 421 |
| 2²⁷ (default) | 30 m off | 11 (11) | 0.1 m | 0.34 | 0 | 773 > 421 |
| 2²⁸ | 30 m off | 15 (15) | 0.07 m | 0.47 | 0 | 1520 > 421 |

- **The threshold.** At 2²⁵ the posterior correctly declines the void, because the data do
  not pay for it.
- **The false void at 2²⁵** sits 3.7 m from chamber 2. The detector rooms (3 m) were
  removed from the unknowns *after* a first false void appeared at 3 m. At 2²⁵ one more
  appears just beyond that radius. It is reported, not re-tuned.
- **Data or prior.** With λ = κ = 0 the mean p(void) over the true void is 0.07. The data
  alone say *that* something is missing along the rays, not *where*. Every found voxel is
  therefore prior-driven, and labelled so.

## Efficiency: the same answer for less work

- **Shorter schedules and fewer branches are not free.** Half the work finds 4 voxels
  instead of 11 (all 4 correct).
- **The thresholded set is noisy in its fringe.** The reference against itself with other
  seeds has a Jaccard index of only 0.36. The p-map is the stable object; the set at
  p ≥ 0.9 is not.
- **Freezing provable rock** (a bound that no context can overturn by 10 nats) removes 24 %
  of the variables in 0.14 s. The 16 branches then take **2.94 s instead of 6.9 s**. They
  find 8 voxels, all correct, with Jaccard 0.73 against the reference, inside the
  seed-to-seed spread.

## Tensors: the ghost bits (tools/branches_blaze.py)

The 20 variables most likely to be void are conditioned on the branch consensus elsewhere
(a declared approximation) and enumerated exactly: 2²⁰ = 1,048,576 branches.

- **Exact local posterior.** All 20 are truly void, at p = 0.987–1.000. The 16-branch
  sampler gave 0.75–1.00, which is why the 0.9 threshold left some out.
- **Blaze TT compression.** The 8 MiB, 20-way tensor becomes TT rank 1–2 with 40 parameters,
  about 26,000× smaller.
  - Marginals read from the TT without rebuilding 2²⁰: error 7 × 10⁻⁷.
  - With int8-quantized cores: error 2.7 × 10⁻³.
- **Why so compressible.** The distribution is sharp: the MAP branch holds 0.974 and three
  states hold 99 %. Given the rest, the bits are nearly independent. A weaker signal would
  raise the ranks, and Blaze's diagnostic would say so.
