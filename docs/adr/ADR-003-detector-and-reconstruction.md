# ADR-003 — The photon-counting detector, and scoring reconstruction against truth

**Status:** accepted
**Date:** 2026-09-11
**Implements:** ADR-001 action items 10–13.

## Context

Phase 1 produces a linear-mode waveform and the truth that made it. Phase 2 showed the
traversal underneath is correct and measured what the RT cores buy.

Neither of those is the product. The product is a dataset an algorithm can be *scored*
on, and two things are still missing: the instrument that actually records single photons,
and a way to say one reconstruction is better than another without appealing to taste.

## Decision

### 1. The detector samples first arrivals, it does not draw per-bin coins

The obvious implementation — for every pulse, for every bin, draw a Poisson count — is
both slow and wrong in an interesting way: it produces a detector with no memory, and the
whole difficulty of single-photon LiDAR is that the detector *has* memory.

A SPAD that fires is blind for a dead time afterwards. So an early photon suppresses every
later one, and the recorded histogram is biased toward short ranges. That is **pile-up**,
and a simulator without it produces data no real detector could produce.

So arrivals are sampled by inverting the cumulative rate. Per pulse:

    target = −ln(1 − u),  u ~ U(0,1)
    find t with R(t) − R(t_live) = target
    record t; t_live = t + τ_dead; repeat until the gate closes

with `R` the cumulative expected count. This gives the exact inhomogeneous Poisson process
with dead time, costs a binary search rather than a pass over every bin, and reduces to
classic TCSPC (at most one photon per pulse) when `τ_dead ≥ gate`.

**Non-paralysable dead time**, not paralysable: photons arriving *during* the dead time are
lost but do not extend it. That is what an actively quenched SPAD does, and the difference
matters at high flux where the paralysable model saturates to zero counts while the real
device saturates to a finite rate.

### 2. The rate is stated in photons, not in watts

Two dimensionless knobs rather than an optical power budget this project cannot honestly
calibrate:

- `signal_photons` — expected detected signal photons per pulse, distributed over the gate
  in proportion to the Phase 1 waveform.
- `ambient_photons` — expected background photons per gate, uniform in time. This is
  daylight, and it is why single-photon LiDAR is hard outdoors rather than in a lab.

Per bin the expected count is then `λ_k = signal·w_k/Σw + ambient/bins`, which is directly
comparable to the linear-mode waveform and needs no unit conversion at the boundary.

### 3. The detector is checked against a closed form, not against "it looks different"

ADR-001 warned that the photon model's real risk is being a scaled copy of the linear one.
Two checks, and the second is the one that has teeth:

- **Dead time off**: `E[n_k]/N = λ_k`. The histogram must converge to the rate.
- **Dead time ≥ gate**: `E[n_k]/N = exp(−R_{k−1}) − exp(−R_k)`, the first-arrival
  distribution. This is analytic, so the pile-up model is checked against *the right
  answer* rather than merely against being different from the no-dead-time case.

And the direction is checkable independently: `n_k / λ_k` must fall monotonically with `k`.
Pile-up cannot make a late bin relatively stronger.

### 4. Reconstruction is scored against truth by an explicit matching, and never by one number

Three baselines, chosen so their failure modes differ:

| | what it is | where it fails |
|---|---|---|
| **peak pick** | the bin with the most counts | cannot see a second surface at all. The control. |
| **matched filter** | correlate with the known pulse, threshold, suppress non-maxima | resolves separated returns; merges close ones; invents returns from noise if the threshold is loose |
| **greedy deconvolution** | repeatedly fit and subtract the strongest pulse | resolves closer pairs than the matched filter; accumulates error into later returns |

Estimated returns are matched to truth returns by **greedy nearest-first assignment within
a declared range tolerance**, one-to-one. What is then reported, separately and never
summed:

- **detection rate** — truth returns that were matched
- **false alarms per beam** — estimates that matched nothing
- **range bias and RMSE** — over matched pairs only

They stay separate on purpose. An algorithm can buy a perfect detection rate by reporting a
return in every bin, and a perfect false-alarm rate by reporting nothing; any single score
that combines them hides which of the two is happening, and the choice of weights would be
doing the arguing.

Bias is reported alongside RMSE because pile-up produces a *systematic* range error, and a
metric that only reports RMSE would describe a ruler that is 15 cm short as merely noisy.

### 5. Coates correction is a baseline, not a fix applied behind the reader's back

The pile-up bias is invertible: given `n_k` over `N` pulses,

    λ̂_k = ln( (N − Σ_{j<k} n_j) / (N − Σ_{j≤k} n_j) )

This is run as a *separate* reconstruction path, so the table shows the same algorithm on
corrected and uncorrected histograms. The measured bias with and without it is the point;
silently correcting everything would hide the effect the detector model exists to produce.

## Consequences

- The per-beam cumulative rate is a prefix sum over the gate. It is computed with a proper
  two-level block scan rather than a serial pass in thread 0 — the serial version is
  correct and would leave 255 of 256 threads idle per beam, which is a bad design knowingly
  embedded rather than a simplification.
- Reconstruction runs on the host. It is per-waveform, cheap, and its correctness matters
  more than its speed; the hot path is the simulator, not the baselines it is scored with.
- Multi-bounce tracing (item 14) is **not** in this ADR. It changes the transport model
  rather than the instrument, and non-line-of-sight imaging deserves its own decision
  record rather than being appended here.

## Action items

1. [x] SPAD sampling: inverted cumulative rate, non-paralysable dead time, ambient
2. [x] Block scan for the cumulative, verified against a host sum
3. [x] Convergence to λ with dead time off
4. [x] Convergence to the analytic first-arrival law with dead time on, and the monotone
       pile-up direction
5. [x] Three reconstruction baselines
6. [x] Greedy matching to truth; detection, false alarms, bias, RMSE
7. [x] Coates correction as its own path, with the bias it removes measured

Measured outcome: `docs/RESULTS-phase3.md`.
