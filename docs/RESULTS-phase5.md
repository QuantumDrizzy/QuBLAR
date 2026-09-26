# Phase 5 results: muon tomography (ADR-006)

Measured on the RTX 5060 Ti (sm_120), CUDA 13.0, with `build\check_muon.exe`. It takes 40 s,
most of it host MLEM over 3 × 25,200 bins.

## What passes

| Check | Result |
|---|---|
| slab optical depth = μL | τ = 1.84 = μL |
| vertical columns vs the voxelized chord | worst relative error 4.0 × 10⁻⁶ |
| detected fraction vs exp(−τ) | worst deviation 2.1 σ |
| open-sky bins vs the cos² law | worst deviation 2.6 σ |
| batch counts vs Binomial(N, T) | variance ratio 0.987 over 860 bins |
| three views deepen the void's deficit | mean μ in the void: 0.0420 /m (3 views) vs 0.0438 /m (1 view) |
| the void reconstructs below the no-void model | 0.0420 vs 0.0463 /m |
| an empty pyramid shows no deficit | 0.0463 /m against rock 0.046 /m |

## What does not, measured and kept

**Localisation of the void by continuous MLEM from three point-like chambers: KNOWN LIMIT.**

The most negative deficit relative to the no-void model sits **7 m above chamber 0**,
30 m from the void. That is the apex artifact of limited-view tomography. Every
void-crossing ray from one chamber also crosses the voxels just above it, and MLEM puts
the deficit where rays converge.

The noise explains why:
- per-voxel noise is about ±0.004 /m;
- the void is air, so a perfect reconstruction would show a deficit of 0.046 /m, but
  MLEM recovers only about 0.003–0.004 /m of it;
- SNR per voxel is about 1.

The void is present in the column profile (0.0435 vs 0.0459 /m at z = 77 m) but is not the
global minimum. It is the baseline that ADR-007's binary reconstruction has to beat.

## Bugs found on the way (each passed or compiled at the time)

- **The MLEM denominator was weighted by N^det instead of N^open.** The code's own comment
  specified N^open. With N^det the fixed point is τ̂ = −ln T / T rather than τ̂ = −ln T, a
  bias of about 1/T (~100× through 100 m of rock). Fixed: numerator and denominator now
  accumulate over the same bins with the same weight.
- **The reconstruction API gained `mu_init` and `domain`, but the check was never updated.**
  It did not compile. Fixed: the prior is the pyramid's surveyed outer shape, solid rock
  inside, and only interior voxels update.
- **The vertical-column check marched the void-carrying medium against a void-free
  reference.** It was off by exactly one void voxel (0.092 = 2 m × 0.046 /m). Fixed: both
  sides use the void-free pyramid.
- **The 48 × 20 sky cannot image the void.** Bins of 7.5° × 3.5° are wider than the 3.7°
  the 2.4 m void subtends at 37 m. The symptom was identical μ from z = 73 to 101 m, one
  ray per column. Fixed: the imaging sky has 1° bins (360 × 70), with 2²⁵ candidates per
  chamber, about 1000 open muons per bin.
- **One point chamber measures directions, not depth.** Fixed: `MuonView`, and MLEM,
  backprojection and illumination over several chambers. There are three: (0, 0, 40),
  (±40, 0, 25).
- **Localisation by the raw darkest voxel lands next to a chamber.** Changed, *after
  seeing it fail*, to the deficit relative to the no-void model exposed with the same
  seeds. That is how real muography reads an anomaly. It still does not localise (above),
  and that is reported rather than re-tuned.
