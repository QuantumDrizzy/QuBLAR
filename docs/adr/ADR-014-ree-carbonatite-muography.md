# ADR-014 -- Exploration muography of a REE carbonatite from an underground drift

**Status:** Proposed -- PRE-REGISTRATION. Sections 1-8 are frozen before any `check_ree_muo` code is written or
run, at the SHA-256 recorded in `experiments/ree_muography/PREREG_SHA256.txt` (unmodified copy:
`experiments/ree_muography/ADR-014.prereg-frozen.md`). Results go in section 9 of the docs copy only.
**Date:** 2026-09-28
**Depends on:** ADR-010 / ADR-011 (Reyna/CSDA survival table, hash-keyed expected maps, Bernoulli cross-check,
profile-likelihood statistic with the density-scale nuisance, template-scan localisation, blind check). The muon
machinery is copied from `src/check_khufu.cu` into a new `src/check_ree_muo.cu`. check_ark and check_khufu are
not modified.

## 0. What this is

A synthetic detectability study. The scene is "Mountain Pass-type" only in its densities and rough scale. It
is not a model of any real mine, and it says nothing about any company's technology. The question: from a drift
200 m below a flat surface, with four 1 m2 muon detectors, which dense REE bodies (size L, depth, density
contrast) are detected, localised and found blind within a stated exposure?

## 1. Sources (context and parameter ranges)

- Deposit context: Mountain Pass carbonatite (Sulfide Queen stock roughly 700 m along strike, roughly 70 m
  thick, dipping roughly 40 deg W; grades to about 15 % REO, average about 8-9 %). This is as summarised by Haxel,
  USGS OFR 2005-1219, after Olson et al., USGS PP 261 (1954), and Castor (2008). The swept bodies below are much
  smaller than the stock: the point is detection limits, not a model of the stock.
- Densities (g/cm3). Host granite/gneiss 2.65-2.75, nominal 2.70 (textbook ranges, e.g. Telford, Geldart &
  Sheriff, Applied Geophysics, 2nd ed., 1990: granite ~2.5-2.8, gneiss ~2.6-3.0). Carbonatite 2.8-3.1. Ore
  grade: modal mix of Sulfide Queen ore of about 65 vol % calcite/dolomite (~2.75), 20-25 % barite / celestine
  / strontianite (~4.3; barite 4.48) and 10-15 % bastnaesite (~4.95) gives ~3.2-3.5. Nominal 3.30.
- Flux sanity references. Mei & Hime, Phys. Rev. D 73, 053004 (2006). Eq. (1), vertical depth-intensity
  relation: I_v(h) = 8.60e-6 e^(-h/0.45) + 0.44e-6 e^(-h/0.87) cm^-2 s^-1 sr^-1, h in km.w.e.; fitted for
  1-10 km.w.e., so it is an EXTRAPOLATION at 0.54 km.w.e., where it gives 2.83e-6. Eq. (4), total flux for flat
  overburden: 67.97e-6 e^(-h/0.285) + 2.071e-6 e^(-h/0.698) cm^-2 s^-1, which gives 1.12e-5 at 0.54 km.w.e.
  (same extrapolation caveat).
- Mining-muography scale references (context, not decisive). Schouten & Ledru, JGR Solid Earth (2018):
  muography for mineral exploration, McArthur River case at ~600 m depth. Ideon Technologies white paper
  (Schouten et al., 2024): borehole/gallery detectors at ~200 m true vertical depth, with density anomalies
  reported at p < 1e-10 after ~1.5 months of data, on ~5 m voxels (vendor document, not peer reviewed).

## 2. Scene (frozen)

Coordinates: x east along the drift, y north, z up; flat surface at z = 0 (air above). Grid 1 m,
x in [-240, 240], y in [-210, 210], z in [-204, 0] (480 x 420 x 204). No ray inside the acceptance leaves the
grid through its side walls before reaching z = 0.
- Host: 2.70 x heterogeneity factor (section 3).
- Drift (air, known, present in every scene and model): x in [-240, 240], y in [-2.5, 2.5], z in [-201, -196].
- Target body: axis-aligned cube of side L centred at (0, 20, -d). Its density is host + drho, with exact
  partial-volume weighting (voxel/cube overlap fraction) and the heterogeneity factor applied to the sum.
- Sweep: depth below surface d in {40, 80, 120, 160} m (the vertical distance from the detectors to the body
  centre is 200 - d = 160, 120, 80, 40 m), L in {5, 10, 20, 40} m, drho in {+0.05, +0.25, +0.60}. That makes 48
  cells:
  - +0.60: ore-grade zone (3.30 vs 2.70);
  - +0.25: carbonatite vs host (2.95 vs 2.70);
  - +0.05: the low-contrast case, EXPECTED TO FAIL (e.g. ~2 vol % bastnaesite replacing calcite, i.e. weakly
    enriched carbonatite against its surroundings).
- K0: host + drift, no body.

## 3. Heterogeneity (nuisance), per seed

Per seed, a smooth relative density field: value noise with N(0, 0.015) node values on a 30 m lattice with a
uniform random offset, trilinearly interpolated. It is multiplied by (1 + 0.03 N(0,1)) iid per voxel. The
realised point s.d. is reported. The analyst's models use the MEAN density (no heterogeneity) with a global
density-scale nuisance s in [0.92, 1.08] (9 grid values, Hermite/golden profile as in ADR-010/011). Seeds 1-8,
with paired backgrounds per seed (K0 and every cell share the seed's field). Blind background seed 101. The
1.5 % / 30 m field is my assumption for unmodelled in-situ variability after the global scale is profiled;
real surveys constrain it with core densities and gravity (declared).

## 4. Detectors and muon model

D1-D4 at (-30, 0, -200), (-10, 0, -200), (10, 0, -200), (30, 0, -200). Each is 1 m2 with box acceptance
|tan theta_x|, |tan theta_y| <= 1 (bin-centre direction). The muon model is exactly as in ADR-010/011: Reyna
2006 sea-level spectrum + CSDA (a = 0.217 GeV/mwe, b = 4e-4 /mwe, p0 = 0.2 GeV/c), cos^2 sky with a 70 deg
zenith cap, 1 x 1 deg bins, 1 muon cm^-2 min^-1 open-sky normalisation. The survival table is extended to
X_max = 1500 mwe. 2^23 hash-keyed rays per detector per scene. Exposure T in {30, 45, 90, 180, 365} days per
detector (45 d ~ the 1.5 months of the Ideon statement).

## 5. Statistics (frozen)

- Detection: ADR-011 s5 statistic over all accepted bins of D1-D4, with weights area x T:
  q = min_s D(d | K0(s)) - min_s D(d | X(s)), where X is the cell's body at its known position.
  Z_emp = (median q_X - mean q_K0) / sd(q_K0), and AUC, over seeds 1-8. Asimov Z (ideal, and with the scale
  nuisance) is reported.
- A cell is "detected" iff Z_emp >= 5 AND AUC >= 0.99 at some T <= 180 d.
- Localisation (template scan): a candidate cube of the cell's L and drho on the grid x in [-60, 60],
  y in [-40, 80], z in [-190, -10], step 2 m. Expected counts N_b(c) = N_b(K0) T(X_b + drho L_b(c)) / T(X_b),
  with X_b the K0 bin-centre opacity and L_b(c) the chord. The estimate minimises sum_b [N_b(c) - d_b ln N_b(c)].
  It uses only data and the K0 model, never the truth. Run for every cell and seed at T = 180 d.
- Minimum detectable size: L_min(d, drho) = the smallest swept L such that it AND every larger swept L are
  detected (T <= 180 d); "none" if L = 40 is not detected.
- Blind: one body with L = 20 m and drho = +0.60 at a hidden centre, uniform in x, y in [-50, 50] and
  z in [-160, -60], rejected unless the centre lies inside the |tan| <= 0.8 window of at least two detectors.
  The seed comes from the OS at run time and is written to disk but not printed. Heterogeneity seed 101,
  T = 180 d. Scan grid x, y in [-70, 70], z in [-190, -10], step 2 m, with the known L and drho template. The
  estimate is written to disk before the truth is revealed.
- MLEM density reconstruction (figure only): cell d = 120, L = 20, +0.60, seed 1, T = 180 d, 2 m grid,
  domain x in [-60, 60], y in [-40, 80], z in [-190, -10], 20 iterations (`mlem_transmission`).

## 6. Decision rules (frozen)

> **R1 (validation / flux scale):** (a) Bernoulli vs expected map, K0 seed 1, D2, 2^24 rays:
> chi2/ndf in [0.8, 1.2]. (b) The simulated open-sky vertical intensity times T(cos = 1, X = 540 mwe) is
> within a factor 2 of Mei & Hime eq. (1) at 0.54 km.w.e. (2.83e-6 cm^-2 s^-1 sr^-1; extrapolated). PASS iff
> both hold. Reported: counts per detector per day in the acceptance, and the simulated total flux (70 deg
> sky) vs eq. (4).
>
> **R2 (ore-grade detection):** drho = +0.60, L = 20 m is detected at all four depths.
>
> **R3 (minimum detectable size vs depth):** prediction, decisive:
> L_min(+0.60) <= 10 m for d >= 120 m, and <= 20 m at every d;
> L_min(+0.25) <= 20 m for d >= 120 m, and <= 40 m at every d. PASS iff all four statements hold.
>
> **R4 (low contrast, expected failure):** drho = +0.05 is NOT detected for any L <= 20 m at any depth. PASS
> iff not detected. L = 40 m is reported.
>
> **R5 (localisation):** every cell detected at T = 180 d is localised within max(5 m, L/2) of the true centre
> in >= 7 of 8 seeds at T = 180 d. PASS iff this holds for all such cells (and at least one exists).
>
> **R6 (blind):** the blind estimate is within 10 m of the hidden centre.
>
> **R7 (literature, reported, not decisive):** our simulated Z at T = 45 d for the ore-grade cells, compared
> in words with the Ideon statement (p < 1e-10, i.e. about 6.4 sigma, after ~1.5 months at ~200 m TVD). Their
> geometry, detectors and anomalies are unknown to me, so this is an order-of-magnitude check.
>
> My prediction: R1 PASS, R2 PASS, R3 PASS, R4 PASS, R5 PASS, R6 PASS. A failure is reported as a failure.

## 7. Declared limits before running

Flat topography; sea-level spectrum (Mountain Pass is at ~1.4 km altitude; ignored); Reyna parametrisation
integrated beyond its stated momentum range; CSDA with constant b, no range straggling or multiple scattering;
point detectors, no resolution, efficiency or cos factor; cube bodies (real ore zones are tabular/irregular);
one heterogeneity model; the detection statistic assumes the body's position (the scan and the blind check do
not); the template uses the true L and drho; 8 seeds; 1 m voxels (partial volume exact for the cube, but the
drift and heterogeneity are voxelised).

## 8. Reproduce

```
nvcc -O3 -arch=sm_120 -std=c++17 -lineinfo -diag-suppress 20013,221 -o build\check_ree_muo.exe src\check_ree_muo.cu
build\check_ree_muo.exe experiments\ree_muography
python tools\render_ree_muo.py experiments\ree_muography
```

## 9. Results

Filled 2026-09-28 after the run. The frozen copy `experiments/ree_muography/ADR-014.prereg-frozen.md` (sha256
6d0f2b58a31c53750e1cdad6b1084b09ae632870432645f142529343cc1a6ab2, frozen 05:08:31+02:00) is unchanged. Run 05:14-05:25,
636.5 s wall time, exit 0. Full tables are in `experiments/ree_muography/RESULTS.md`.

- **R1 FAIL (flux clause).** (a) Bernoulli K0 seed 1, D2, 2^24 rays: chi2/ndf 1.021 (ndf 2107) PASS (exploratory 2^28:
  1.002). (b) Simulated vertical intensity at 540 mwe is 7.56e-6 cm^-2 s^-1 sr^-1 vs Mei & Hime eq. (1) 2.83e-6
  (extrapolated): ratio 2.67 > 2, FAIL. The total flux vs eq. (4) is 1.39e-5 vs 1.12e-5 (1.24). K0 gives 9 613 counts per
  1 m2 detector per day in the acceptance.
- **R2 PASS.** The +0.60, L = 20 cube is detected at every depth. Z_emp at 180 d: 6.39 / 8.46 / 11.32 / 11.27 (d = 40 / 80 /
  120 / 160, AUC 1.00); first detected at 45 / 45 / 30 / 30 d.
- **R3 FAIL.** L_min(+0.60) = 20 / 20 / 10 / 10 m, so both ore statements hold. L_min(+0.25) = 40 / 40 / 40 / 20 m, so
  "<= 20 m for d >= 120" fails at d = 120 (Z_emp 4.70 at 180 d, 3.93 at 365 d). L_min(+0.05) = none.
- **R4 PASS.** +0.05 is never detected for L <= 20 (max Z_emp 1.76); L = 40 is not detected either (max 3.49, AUC 0.98).
- **R5 FAIL.** 13 of the 15 cells detected at 180 d are localised within tolerance in >= 7/8 seeds (median error 0-4 m).
  Two fail: d 120 / L 10 / +0.60 (4/8, median 27.7 m) and d 160 / L 10 / +0.60 (5/8, median 4.2 m vs 5 m tolerance).
- **R6 PASS.** Blind L 20 / +0.60 body at (32.03, 13.89, -86.93). The estimate (32, 14, -86) was written first; error 0.93 m.
- **R7 (reported).** Ore-grade L = 20 at 45 d: Z_emp 6.0-8.3. This is the same order as the Ideon white paper's p < 1e-10
  after ~1.5 months at ~200 m TVD.
- Prediction check: all six PASS predicted; R1, R3 and R5 failed. The limiting factor is the in-situ heterogeneity
  (0.8 % smooth density field), not counting statistics. Z_emp is nearly flat from 30 to 365 d, while the Asimov Z
  (scale nuisance only) grows as sqrt(T): 22.3 at 180 d for the ore L = 20 cube at d = 120.
- If the true flux is the Mei & Hime eq. (1) value (2.67x lower), 180 d of real exposure corresponds to our ~67 d. All
  ore-grade L = 20 cells are already detected at 45 d and 90 d, so R2 would not change. This is a scaling argument, not a run.
