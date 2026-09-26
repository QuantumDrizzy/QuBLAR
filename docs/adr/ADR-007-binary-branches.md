# ADR-007: Binary branches — finding what is not there, as an Ising/QUBO problem

**Status:** accepted
**Date:** 2026-09-26
**Follows:** ADR-006 (muon mode). Same data, same forward model, same truth. What
changes is the question asked of the data.

## Context

ADR-006 ended on a measured limit. Continuous MLEM from three point-like chambers
recovers the Big Void's deficit only at the noise level (≈ 0.003 /m against ±0.004 /m),
and puts its peak at the apex above a chamber, 30 m off.

A continuous reconstruction asks *how dense is every voxel*. With about 75,000 rays and
about 300,000 interior voxels, that question is massively underdetermined, and the
answer is smeared along the rays.

The owner's reframing is a game of hide and seek with bits. Every voxel is a bit:
**rock (exists)** or **void (does not exist)**. The void is not a region of low density
to estimate; it is an *absence* to find, and absence is a datum. The reconstruction must
render three states:

| state | meaning | rendered as |
|---|---|---|
| exists | every branch says rock | rock |
| does not exist | every branch says void: an absence the data demand | the found void |
| ~~exists~~ | branches disagree: the data cannot decide | struck through, uncertain |

**Branches**, in the owner's Everett vocabulary with the ontology removed, are complete
bit assignments that the data do not rule out. Many independent anneals produce many
branches. Where they agree, the bit is decided; where they disagree, it is not. Nothing
here is quantum. The problem is a QUBO; a quantum annealer is one possible solver and
is out of scope.

"Bits are dimensions": N binary voxels form one point of the N-dimensional hypercube
{0,1}ᴺ. Every branch is a vertex, and a single flip moves along one edge. Pixels, or the
rendered voxels, are how a human sees one vertex in 3D.

## Decision

### 1. The energy (a QUBO)

Voxel v in the reconstruction domain carries x_v ∈ {0,1}, with 1 = void, so
μ_v = μ_rock(1 − x_v). For bin b of view c, the no-void model gives optical depth τ⁰_b,
and the data give t_b = −ln(N^det_b / N^open_b). The deficit the data demand is
d_b = τ⁰_b − t_b, and the deficit a configuration explains is Σ_v a_bv x_v, with
a_bv = μ_rock·ℓ_bv.

```
E(x) =  Σ_b w_b (Σ_v a_bv x_v − d_b)²      data; w_b = N^det_b ≈ 1/Var(t_b)
     +  λ Σ_<uv> [x_u ≠ x_v]              walls are continuous (6-neighbour)
     +  κ Σ_v x_v                          voids are rare
```

All three terms are quadratic in binary x. The Poisson weight w_b comes from
Var(−ln T̂) ≈ 1/N^det, derived, not tuned. λ and κ are declared per experiment, and
every result states them.

### 2. The solver: sparse simulated annealing on the host, in C++

- Each voxel keeps its ray list (bin, a_bv) in CSR form, about 7.5 M entries for three
  views.
- A flip updates only the residuals of the rays through that voxel, so ΔE is local.
- The schedule is geometric cooling, with the temperatures declared.
- **The oracle:** exhaustive enumeration of a 16-voxel toy (2¹⁶ configurations). The
  annealer must reach its ground-state energy before any pyramid number is reported
  (the rule of DRIFT and ndim-lab).
- A CUDA path comes only after the host path is correct, and must match it.

### 3. Branches and the tri-state map

- R independent anneals (seeds 1…R) are R branches.
- p_v = the fraction of branches with x_v = 1.
- **Exists** if p_v ≤ 0.1, **does not exist** if p_v ≥ 0.9, **~~exists~~** otherwise. The
  thresholds are declared, and calibration is checked against truth (§5).
- **Data or prior.** The run is repeated with λ = κ = 0. A voxel decided only when the
  prior is on is labelled **prior-driven**. The render marks it, so the engine never
  passes off its assumptions as seen.

### 4. Blaze: compressing the branches

The branch ensemble is R vectors of N bits. The object behind it is a distribution over
2ᴺ configurations, which is exactly the high-order tensor Blaze compresses: one mode
per bit, so each bit is a dimension. For a region of interest of n ≲ 40 voxels around a
detection:

- the empirical branch distribution is a 2ⁿ tensor;
- Blaze gives it a TT/MPS form, and its marginals and overlaps are computed **in the
  compressed form** (Blaze Phase 7, O(nχ³)).

Blaze's own contract applies: TT compresses only TT-native structure. If the branch
distribution is not low-rank across the cuts of the voxel ordering, Blaze reports it, and
so does this repository. This runs as tooling (Python, Blaze's API) over exported
branches, as ADR-001 allows. The hot path stays C++/CUDA.

### 5. Scoring against truth (the QuBLAR rule)

- **Void found:** IoU of the "does not exist" voxels with the true void, and the distance
  from their centroid to the void's axis.
- **No hallucination:** the empty pyramid, exposed with the same seeds and solved with
  the same λ and κ, must yield **no** "does not exist" voxels.
- **Calibration:** among voxels with p_v in a band, the fraction truly void must match
  the band.
- **Baseline:** ADR-006's continuous MLEM on the same data. The binary result is reported
  next to it, whichever wins.

## Falsifiers

- The annealer misses the 16-voxel ground state.
- The empty pyramid produces confident voids: hallucination, and the build fails.
- A voxel is certain only through the prior and is not marked prior-driven.
- The binary reconstruction localises the void no better than MLEM. This is reported
  either way; it is the question.

## Consequences

- New files: `src/ising_recon.hpp` (the energy, CSR, annealer, branches) and
  `src/check_ising.cu` (oracle, void, empty control, calibration), wired into build,
  check and sanitize.
- `tools/export_branches.py` and `tools/branches_blaze.py` (Blaze) are tooling over
  exported files.
- Rendering the tri-state map in 3D is ndim-lab's job, on the exported `.npz`.
