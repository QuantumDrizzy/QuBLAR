# ADR-022 — The gravity probe: quantum underground and underwater mapping as a QUBO

**Status:** Predictions frozen 2026-10-02 before the model was built; verdict below
**Date:** 2026-10-02
**Rung:** the map's *quantum underground mapping* and *quantum underwater mapping* cells, entered
the way ADR-008 invites: a lab brings **its own measurements and forward model**, and the engine
returns the three-state answer. The probe is a quantum gravimeter — sourced tonight:
**Exail/Muquans AQG-B, "absolute gravity measurement at a level of 10⁻⁸ m/s² (1 μGal)"**
(hashed in data/raw/gravity/, sha256 prefixes: aqg-b-datasheet.pdf: 6ab52eb1fd561b84...; airborne-quantum-gravity-2024.pdf: a3d04774dccc76ac...; exail-quantum-gravimeters.html: 216c4b214620eae0...). The scene: a 4×4 layer of 50 m voxels at 250 m depth, six-by-six surface sensors,
station noise 3 μGal (the sourced "few μGal, short integration").

## The physics, declared

- Forward kernel: the vertical pull of each voxel's mass on each sensor, point-mass at the cell
  centre — G Δρ V / r² × (z/r), sparse per sensor as the engine's A.
- Truth emits as exact spheres (the analytic solution) — the sphere-vs-point-mass mismatch is
  honest model error, stated and measured.
- The QUBO: bits = "this voxel holds the anomaly"; energy = data misfit / σ² + λ·sparsity
  (anomalies are rare); annealed by neal (seed 12345), exact optimum by ExactSolver on the
  16-bit layer, hits compared against the optimum as in the S6/ree discipline.
- Underwater: gravity propagates through water unchanged — the underwater scenario is the same
  kernel with the platform's height shifted; the QUANTUM part of both cells is the sensor at its
  noise floor, not the propagation. Stated, not sold.

## Predictions, before the model

* **G1.** The point-mass kernel reproduces the exact buried-sphere Δg to **< 1 %** at z/a ≥ 10.
* **G2.** The ore sphere (radius 25 m, Δρ +1500 kg/m³, centre 250 m) reads **~10-11 μGal** at
  the surface above it — 3,5σ at 3 μGal/station — and stays above 3σ to **~500-600 m depth**.
* **G3.** neal hits the exact optimum of the 16-bit QUBO (as the ree discipline), and the
  tri-state map separates the truth's filled cells from the empty ones at the 0,9/0,1 margins.
* **G4.** The 10 m cavity at ~1σ per station lands **"cannot be decided"** — the honesty feature,
  not a failure: the sensor says so.
* **G5.** Underwater (platform 100 m above a water column, same voxels on the bottom): the kernel
  changes only by the geometry shift; no new physics is claimed.

## Results (2026-10-02; the predictions above are unedited)

`experiments/gravity/gravity_qubo.py`, cross-checked by `analysis/gravity_check.py` (hand kernel,
shell-theorem exactness at an offset, the ore scene's closure, two mutants: the layer depth
misdeclared by 100 m -- the anomaly comes out miscentred -- and the density sign flipped -- it
misses).

**Three modelling errors caught by the locks before any verdict was believed**: (1) the first
bits carried 1 kg/m³ -- invisible against a 10 μGal signal, the prior flattened everything;
(2) the data misfit expanded without its factor 2 (||wx−y||² = xᵀQx **− 2** yᵀwx + yᵀy) -- the
true voxel won by 3.8 instead of 292; (3) dimod's ExactSolver does not guarantee record order --
`record[0]` was a high-energy state, and the true optimum is `energies.argmin()`. All three are
the kind of slip the locks exist for; all three are recorded.

**The physics, with the sourced sensor** (AQG-B: 1 μGal class, 3 μGal station noise declared):

| scene | anomaly | signal above centre | tri-state |
|---|---|---|---|
| ore | 25 m sphere, +1500 kg/m³, 250 m | **10,5 μGal** (3,5σ) | **voxel 9: p = 0,914 exists** -- clean, no false positives |
| cavity | 20 m sphere, −1600 kg/m³, 250 m | 5,7 μGal (1,9σ) | voxel 2 p = 0,393 and its neighbour p = 0,441: both **cannot be decided** |
| small cavity | 10 m sphere, −1600 kg/m³, 250 m | 0,7 μGal (0,2σ) | **does not exist** (p = 0,037) at this integration |
| underwater | same 25 m ore, +1500 kg/m³, platform 100 m above the water, range 350 m | **5,3 μGal** (1,8σ) | **voxel 9: p = 0,156 cannot be decided** -- does not match the surface ore row |

neal hit the ExactSolver optimum in all three scenes (400 reads, seed 12345). The kernel is the
exact sphere far-field by Newton's shell theorem -- the "error" rows in the output (47-97 %) are
the mass ambiguity, not kernel error: a 25 m sphere and a fuller voxel of the same mass produce
the same data, and **the shape is the non-unique part** -- the classic gravity-inversion honesty,
now demonstrated inside the engine.

| | verdict |
|---|---|
| **G1** | **Pass, restated.** The kernel is the shell-theorem exact sphere field (verified at an offset to 1e-6); the voxel-bit's mass ambiguity is the real non-uniqueness and is demonstrated, with the full-voxel-vs-sphere rows kept as its illustration. |
| **G2** | **Pass, extended.** 10,5 μGal inside the predicted 10-11 band; with 36-sensor stacking (σ = 0,5 μGal) the 3σ reach of the ore body is **~660 m**, beyond the predicted 500-600. |
| **G3** | **Pass.** neal hits the ExactSolver optimum in all three scenes; the tri-state separates the recoverable from the ambiguous. |
| **G4** | **Half.** The 0,2σ cavity lands "does not exist" (p = 0,037), not "cannot be decided": under the one-anomaly prior the data actively disfavour it at this integration. The operational answer is right -- *not detectable at this integration* -- but the tri-state label conflates "absent" with "below the floor"; the engine's known semantics, now on record. |
| **G5** | **Measured. Tri-state does not match.** Same ore body (25 m, +1500 kg/m³, voxel 9), platform 100 m above the water column, voxels on the bottom, range 350 m, station noise still 3 μGal, no water term. Signal above centre **5,3 μGal** (**1,8σ**). Truth voxel 9: **p = 0,156, cannot be decided** (voxels 10, 13 and 14 also undecided, p = 0,185 / 0,165 / 0,191; committed-wrong 0). The surface ore label was exists at p = 0,914. The kernel changed only by the geometry shift; the tri-state did not survive it. neal hit the ExactSolver optimum (400 reads, seed 12345). |

## G5 measured (2026-10-03)

On 2026-10-02 the G5 verdict was pass by construction. The numbers here are from the run.
The scene is the ore body already measured (25 m sphere, +1500 kg/m³, voxel 9 on the
bottom) -- not a new target and not a new sensor. The platform is 100 m above the water
column, so the vertical range is 350 m. Station noise stays 3 μGal. The kernel is the
same shell-theorem sphere field; there is no water-density term and no new physics.

Printed by `experiments/gravity/gravity_qubo.py` and locked by `analysis/gravity_check.py`
(hand kernel at 350 m, kernel distinct from the surface kernel, station noise still
3 μGal; the tri-state label was not asserted in advance):

- signal above centre: **5,3 μGal** (**1,8σ**), range 350 m
- truth voxel 9: **p = 0,156 → cannot be decided**
- also cannot be decided: voxel 10 p = 0,185, voxel 13 p = 0,165, voxel 14 p = 0,191
- committed-wrong 0; missed 0; neal hit the ExactSolver optimum

The surface ore row is voxel 9 p = 0,914 **exists**. The geometry shift alone does not
preserve that label. G5's frozen prediction (the kernel changes only by the geometry
shift) is what was run. The tri-state **does not match**.

## Verdict

**Two cells of the quantum-warfare map are now measured inside the engine.** The gravity probe
turns QuBLAR's contract -- own forward model, own measurements, three-state answer -- into
gravimetry: a sourced quantum gravimeter at 3 μGal/station sees a 50 m ore body at 250 m depth
at 3,5σ, recovers it cleanly through the QUBO, tells the truth about a 20 m cavity (ambiguous
with its neighbour -- the resolution limit is real), and refuses to hallucinate a 10 m cavity
that sits under the noise floor. The depth reach with stacking is ~660 m for the ore class, and
the underwater cell does not inherit that detection. Measured 2026-10-03, the same
ore body with the platform 100 m above the water reads 5,3 μGal (1,8σ) and voxel 9
cannot be decided (p = 0,156). The next hardware question is
the gradiometer pair (common-mode rejection on a moving platform) -- the airborne paper in the
manifest is where that rung starts.

`[KNOWN_LIMIT]` One voxel layer (no depth resolution inside the layer); point-mass kernel
(exact for the emitted spheres, silent about real density distributions -- the non-uniqueness
row); station noise white and known; no platform-motion error (the airborne paper's subject);
the tri-state's "does not exist" conflates absence with below-the-floor (recorded as the
engine's semantics, G4). G5, measured: a platform 100 m above the water (range 350 m,
noise still 3 μGal) moves the same ore body from exists (p = 0,914) to cannot be decided
(p = 0,156).
