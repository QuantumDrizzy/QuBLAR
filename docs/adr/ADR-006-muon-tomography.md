# ADR-006 — Muon mode: replicating ScanPyramids synthetically

**Status:** accepted
**Date:** 2026-09-18
**Follows:** ADR-005 (external validation). Uses the same discipline: a transport
model, a counting process, truth emitted alongside the measurement, and
reconstructions scored against that truth.

## Context

Phase 4 closed the loop with real external data. This phase opens a second
front: the same pipeline shape -- probe, count arrivals, reconstruct, score --
applied to the probe that found the Big Void. Cosmic-ray muons pass through
hundred metres of limestone where photons stop at centimetres; ScanPyramids
located a ≥30 m void above the Grand Gallery of Khufu's pyramid by imaging the
shadow it casts in the muon flux (Morishima et al., Nature 552, 2017). A
synthetic replication is a demonstration, not a discovery -- but it is the
demonstration this repository exists for: every claim scored against truth the
simulator emits, which no real campaign can do.

### What was measured before deciding

| | |
|---|---|
| Integrated sea-level muon flux | ≈ 70 m⁻² s⁻¹ ≈ 1 cm⁻² min⁻¹ (the standard figure; used to sanity-check exposure scales, not as a fitted parameter) |
| Attenuation through limestone | A ~GeV muon loses ≈ 0.4 GeV per 100 g/cm²; 100 m of rock costs ≳ 110 GeV, so the detected population is the high-energy tail. Real muography folds the energy spectrum into an effective transmission curve; T(100 m) ≈ 10⁻²–10⁻³. |
| ScanPyramids numbers used for the replica | Pyramid height 139 m, base 230 m; void ≥30 m long, cross-section ≈ 2×2 m, directly above the Grand Gallery (≈ z 75–80 m); detectors in chambers at ≈ 40 m elevation; months of exposure. |

## Decision

### 1. Transmission, not energy loss

The muon is not transported through an energy-loss integral. A muon that enters
the medium along ray r is **detected with probability exp(−τ_r)**, with the
optical depth τ_r = ∫ μ(s) ds and μ = μ_eff·(ρ/ρ_rock). This is the
Beer-Lambert model every transmission muography analysis reduces to, with the
energy spectrum folded into one declared effective coefficient
**μ_eff = 4.6×10⁻² m⁻¹ of rock** (T(100 m) ≈ 1%, matching the published
transmission scale). The honest sentence: this simulates the *counting
statistics of shadow imaging*, not the cascade physics inside it.

### 2. A point-like chamber, a binned sky, and truth in three layers

ScanPyramids' emulsion stack in one chamber subtends tens of centimetres
against a 100 m pyramid: the chamber is treated as a **point** at (0, 0, 40),
so a backward ray is fully determined by its direction. Candidates are sampled
from the sky model -- **dN/dΩ ∝ cos²θ**, azimuth uniform, θ ≤ 70°, sampled in
closed form via cosθ = (c_min³ + u(1−c_min³))^⅓ -- and binned into a
48 × 20 (azimuth × zenith) histogram. The data product is per-bin
open/detected counts, exactly the angular histogram a real analysis uses.

Truth is emitted at three levels and each is checked against a different kind
of answer: the **medium itself** (the density map is the ground truth), the
**per-ray optical depth** (checked against analytic chords), and the
**binned counts** (checked against the Binomial closed form below).

### 3. The marcher is shared by transport and reconstruction

Optical depth is accumulated by an Amanatides–Woo traversal of the uniform
voxel grid, written `__host__ __device__` once: the GPU exposes candidates with
it, and the host reconstruction re-marches bin-centre rays with the same code.
Two implementations of one marcher is the duplication that cost Phase 2 a
confound; there will not be two.

### 4. Transmission MLEM, derived rather than copied

Per bin b, N^det ~ Binomial(N^open, e^{−τ̂_b}) with τ̂_b = Σ_v μ_v ℓ_{b,v}.
The multiplicative update implemented is

    μ_v ← μ_v · Σ_b ℓ_{b,v}·N^open_b·e^{−τ̂_b} / Σ_b ℓ_{b,v}·N^det_b

chosen because its fixed point is exactly e^{−τ̂_b} = N^det/N^open and its
direction is right: a voxel set too high makes τ̂ too big, the numerator too
small, and itself smaller. The control is a **one-step least-squares
backprojection** (μ̂_v = Σ_b ℓ_{b,v}(−ln T_b) / Σ_b ℓ_{b,v}², N^open-weighted)
-- the oldest inversion that assumes nothing beyond the line-integral model.

### 5. Detection is Bernoulli, and that is the checkable statistics

Candidates are hash-sampled (deterministic per index, the ADR-001 rule), so a
fixed candidate population resampled under detection alone gives
**Binomial(N, T) batch statistics** -- checked against its closed form, which
is a sharper check than "looks Poisson" and doubles as a determinism check on
the sampler.

### 6. What is not modelled, named so it cannot be forgotten

- **Multiple scattering** (mrad-level; below the 2 m voxel at 40 m standoff,
  but it is why real angular histograms have finite pointing resolution).
- **The energy spectrum** -- folded into μ_eff (§1); a real analysis bins by
  scattering angle to recover it.
- **The known chambers** (King's, Queen's, Grand Gallery): the replica is solid
  rock plus the one void. A fuller replica would carry them as background
  structure, which is exactly what makes the real inverse problem hard.
- **Detector pointing jitter and alignment error** -- the things that eat real
  campaigns; absent by construction here.

### 7. What the numbers may say

The analytic checks (slab, voxelized chord, sky law, Binomial dispersion)
validate the transport. The replica localisation validates MLEM and the
control against the transport model. Neither says anything about the real
pyramid beyond what the declared parameters put in: the void is where it is
because it was placed there. Any sentence beginning "this shows ScanPyramids
is right" is out of scope -- the sentence this phase supports is "this is how
the measurement works, end to end, with the answer known".

## Action items

1. [x] `src/muon.cuh` -- voxel medium, shared marcher, sky sampler, exposure kernel
2. [x] `src/muon_recon.hpp` -- binned transmission, MLEM, one-step backprojection, scoring
3. [x] `src/check_muon.cu` -- slab; voxelized pyramid chord; cos² law; Binomial
       dispersion; void localisation; empty control
4. [x] build/check/sanitize integration
5. [x] `docs/RESULTS-phase5.md` after measurement

Measured outcome: `docs/RESULTS-phase5.md`.

## Amendment (2026-09-26): what building it changed

- **Several views, not one.** §2's single point-like chamber measures directions only.
  Reconstruction takes `MuonView`s, and the replica uses three chambers.
- **The imaging sky has 1° bins.** The 48 × 20 sky stays for the statistics checks (C and
  D), and cannot resolve a 2.4 m void.
- **§4's update is now the bounded log-transmission form in `muon_recon.hpp`**, with
  numerator and denominator weighted by N^open over the same bins, a surveyed-shape prior
  and an update domain.
- **§7 held.** Continuous MLEM from three chambers does not localise the void. That is
  recorded as a KNOWN LIMIT in RESULTS-phase5, not tuned away.
