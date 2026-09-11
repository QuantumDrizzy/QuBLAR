# Phase 3 result — photon counting, and reconstruction scored against truth

Measured 2026-09-11. RTX 5060 Ti, CUDA 13.0, Windows toolchain. Scene: two planes at
10 m and 12 m meeting at a range discontinuity, scanned by 96 beams of 2 mrad divergence
so that 20 of them straddle the edge and see two surfaces. 2048 bins of 0.2 ns, 1 ns
pulse FWHM.

## The detector is checked against closed forms, not against looking plausible

ADR-001 named the risk: a photon-counting model that is really the linear waveform wearing
noise. Two exact laws settle it, and they are different from each other, so no scale factor
satisfies both.

| | expected | measured |
|---|---|---|
| dead time off | `E[n_k]/N = λ_k` | max \|z\| = 3.51 over **2048** bins |
| dead time ≥ gate | `E[n_k]/N = exp(−R_{k−1}) − exp(−R_k)` | max \|z\| = 3.03 over **341** bins |

Agreement is judged by z-score against the known multinomial/Poisson variance rather than
against a tolerance, so there is no epsilon here that could have been tuned until the test
passed.

**The bin counts are themselves the result.** With nothing blocking, every bin in the gate
carries enough counts to be tested. With dead time on, only 341 do — once the detector has
fired, `exp(−R)` collapses and the rest of the gate stops accumulating statistics at all.
That drop from 2048 to 341 is pile-up showing up in a second, independent way, and it is
asserted rather than merely observed.

## Reconstruction, three baselines, three inputs

Range tolerance for a match: 30 cm. Detection, false alarms, bias and RMSE are reported
separately and never combined into a score — an algorithm can buy a perfect detection rate
by reporting a return in every bin.

| input | algorithm | detect | false/beam | bias m | rmse m |
|---|---|---:|---:|---:|---:|
| linear waveform | peak pick | 82.8% | 0.00 | +0.0021 | 0.0048 |
| | matched filter | 94.0% | 0.00 | +0.0022 | 0.0048 |
| | greedy deconvolve | 94.0% | 0.00 | **+0.0000** | **0.0001** |
| photon counts | peak pick | 82.8% | 0.00 | −0.0393 | 0.0409 |
| | matched filter | 93.1% | 0.00 | −0.0263 | 0.0272 |
| | greedy deconvolve | 93.1% | 0.00 | **−0.0332** | 0.0344 |
| counts + Coates | peak pick | 82.8% | 0.00 | +0.0021 | 0.0048 |
| | matched filter | 94.0% | 0.00 | +0.0022 | 0.0048 |
| | greedy deconvolve | 94.0% | 0.00 | **−0.0000** | 0.0009 |

Three things worth reading off it:

**Pile-up costs 3.3 cm of range, systematically short.** Not noise — bias. A metric
reporting only RMSE would have described this as a slightly noisy instrument rather than a
ruler that is consistently wrong in one direction.

**Coates correction removes essentially all of it**, −3.3 cm → −0.001 cm, and restores the
detection rate to the noiseless figure. It is run as its own path rather than applied
everywhere, so the size of the effect stays visible.

**The single-return control earns its place.** Peak pick tops out at 82.8% because 20 of
the 96 beams see two surfaces and it can only ever report one; 96 of 116 truth returns is
exactly 82.8%. The multi-return methods reach 94.0%. Had they not beaten it, the
elaboration would not have been paying for itself.

## Where the baselines break

The table above reports zero false alarms on every row, which is not a result — it says the
scene was too easy to tell the methods apart. Sweeping the photon budget at 1.0 ambient
photons per gate, 20000 pulses, greedy deconvolution on raw counts:

| signal/pulse | detect | false/beam | bias m | rmse m |
|---:|---:|---:|---:|---:|
| 0.0005 | **0.0%** | 0.00 | +0.0000 | 0.0000 |
| 0.0020 | **0.0%** | 0.00 | +0.0000 | 0.0000 |
| 0.0100 | 85.3% | 0.00 | −0.0007 | 0.0076 |
| 0.0500 | 93.1% | 0.07 | −0.0017 | 0.0042 |
| 0.5000 | **94.8%** | 0.18 | −0.0090 | 0.0096 |
| 2.0000 | 93.1% | 0.00 | −0.0332 | 0.0345 |
| 10.0000 | 87.9% | 0.00 | **−0.0889** | 0.0907 |

Both ends fail, for unrelated reasons. At the bottom the detector is starved and nothing
clears the threshold. At the top pile-up dominates and the ranges come back 9 cm short
while detection falls as returns merge into the leading edge. The usable band is in
between, and false alarms are highest not at either extreme but at 0.05–0.5 photons per
pulse, where the background is strong enough to produce peaks and the signal is not yet
strong enough to dominate them.

A first attempt at this sweep ran 0.02 to 10 photons per pulse at 0.3 ambient and found
94% detection at the bottom. That was not a robust detector, it was a badly chosen axis:
0.02 photons per pulse over 20000 pulses is still 400 photons. What starves the instrument
is signal against background at a fixed acquisition time.

## A race the invariants could not see

`racecheck` reported 6 hazards in the detector kernel, and they were real. Every thread
reads `cum[bins-1]` to get the total returned energy, then the threads rewrite `cum` in
place to rescale it into photons — with no barrier between the read and the writes. The
thread owning the last bin can store its rescaled value while another thread is still
reading that slot, and that thread then normalises its share of the cumulative against a
corrupted total.

Nothing above would have caught it. The histogram stays entirely plausible; only the photon
budget is quietly wrong, by an amount that varies with scheduling. One `__syncthreads()`.

## Status

memcheck 0 errors, racecheck 0 hazards, all suites green. ADR-001 items 1–13 are done.
Item 14 — multi-bounce tracing, the door to non-line-of-sight imaging — is not, and
deliberately: it changes the transport model rather than the instrument, and deserves its
own decision record.
