# ADR-014 results: exploration muography of a REE carbonatite from a drift (SYNTHETIC)

Pre-registration: `ADR-014.prereg-frozen.md`, SHA-256
`6d0f2b58a31c53750e1cdad6b1084b09ae632870432645f142529343cc1a6ab2`, frozen 2026-09-28T05:08:31+02:00 (git HEAD
2747e94), before any ADR-014 code existed. The hash was re-checked after the run. The scene is Mountain-Pass-like only
in its densities and scale; it says nothing about any real mine or company.

## Commands (PowerShell, repo root)

```
cmd /c '"...\vcvars64.bat" >nul 2>&1 && nvcc -O3 -arch=sm_120 -std=c++17 -lineinfo -diag-suppress 20013,221 -o build\check_ree_muo.exe src\check_ree_muo.cu'
build\check_ree_muo.exe experiments\ree_muography 23 8 1      # 2^23 rays/detector/scene, seeds 1-8, MLEM on; 636.5 s wall
$env:PYTHONDONTWRITEBYTECODE=1
python tools\ree_muo_rules.py experiments\ree_muography        # -> ree_muo_rules.json (rule verdicts)
python tools\render_ree_muo.py experiments\ree_muography       # -> figures\ (PNG + orbit MP4), ~45 s
```

Smoke test before the run (not a result): `build\check_ree_muo.exe build\smoke_muo 20 2 1` (2^20 rays, 2 seeds).
Run: started 05:14:12, ended 05:24:49 (+02:00), exit 0.

Seeds:
- heterogeneity seeds 1-8 (paired: K0 and all 48 cells share each seed's field); blind background seed 101;
- Poisson draw tags are fixed in the code (statistic `0x524545*1000003 + 1009(k+1) + 17e`; localisation `0x10CA14*7919 + 101c + k+1`;
  blind `0xB11D14`); Bernoulli seed 777;
- the blind position seed came from the OS at run time (`blind_seed.txt`). It was not printed, and the estimate was written before the truth.

## Setup actually run

- Grid 1 m, 480 x 420 x 204 (x in [-240, 240], y in [-210, 210], z in [-204, 0]), flat surface at z = 0.
- Host 2.70 g/cm3; drift 5 x 5 m.
- Four 1 m2 detectors at (-30/-10/10/30, 0, -200), box acceptance |tan| <= 1 (17 328 one-degree bins each; 69 312 in the statistic).
- Heterogeneity realised: smooth-field point s.d. 0.80 %, total per-voxel s.d. 3.1 % (the iid part dominates per voxel).
- 48 cells: cubes of L in {5, 10, 20, 40} m at (0, 20, -d), d in {40, 80, 120, 160} m, drho in {+0.05, +0.25, +0.60}.
- Exposures 30 / 45 / 90 / 180 / 365 d per detector.

## Frozen rules and verdicts

| rule | verdict | numbers |
|---|---|---|
| R1 validation / flux | **FAIL** (flux clause) | (a) Bernoulli K0 seed 1, D2, 2^24 rays: chi2/ndf = 1.021 (ndf 2107), expected 11 198 vs detected 11 269: PASS. Exploratory 2^28: 1.002 (ndf 15 584). (b) Simulated vertical intensity at 540 mwe is 7.56e-6 cm^-2 s^-1 sr^-1 vs Mei & Hime eq. (1) 2.83e-6 (extrapolated below its 1-10 km.w.e. range). Ratio 2.67 is outside [0.5, 2]: FAIL. Reported: simulated total flux (70 deg sky, flat overburden) 1.39e-5 vs eq. (4) 1.12e-5 (ratio 1.24, also extrapolated). K0 counts per 1 m2 detector per day in the acceptance: 9 613 (about 1.7e6 in 180 d). |
| R2 ore-grade detection (+0.60, L = 20) | **PASS** | Z_emp at 180 d (AUC 1.00): d 40: 6.39, d 80: 8.46, d 120: 11.32, d 160: 11.27. First detected at T = 45 / 45 / 30 / 30 d. |
| R3 minimum detectable size | **FAIL** (1 of 4 statements) | L_min(+0.60) = 20 / 20 / 10 / 10 m at d = 40 / 80 / 120 / 160: both ore statements hold. L_min(+0.25) = 40 / 40 / 40 / 20 m: "<= 40 m at every d" holds, but "<= 20 m for d >= 120" fails at d = 120. There the carbonatite L = 20 cube reaches only Z_emp 4.70 (AUC 1.00) at 180 d, and 3.93 at 365 d. L_min(+0.05) = none at every depth. |
| R4 low contrast not detected (L <= 20) | **PASS** | No +0.05 cell with L <= 20 detected (max Z_emp 1.76, AUC 0.83, d 120 L 10 at 180 d). L = 40 at +0.05 is not detected either (max Z_emp 3.49, AUC 0.98, d 160, 90 d). |
| R5 localisation of cells detected at 180 d | **FAIL** (13 of 15) | 15 cells detected at 180 d. 13 localised within max(5 m, L/2) in >= 7/8 seeds (median errors 0-4 m). Fails: d 120 / L 10 / +0.60 at 4/8 (median error 27.7 m) and d 160 / L 10 / +0.60 at 5/8 (median 4.2 m; tolerance 5 m). In the failing seeds the scan lands 5.7-74 m away, on heterogeneity-made minima. |
| R6 blind | **PASS** | Hidden centre (32.03, 13.89, -86.93); estimate (32, 14, -86), written first; error 0.93 m (tolerance 10 m). The body is L 20, +0.60, about 113 m above the drift; 1 rejection try. |
| R7 literature (not decisive) | reported | Ore-grade L = 20 cubes at T = 45 d: Z_emp 6.0 / 6.2 / 8.3 / 6.8 (d = 40 / 80 / 120 / 160); L = 40: 13.7-27.0; L = 10: 1.5-2.3. The Ideon white paper reports p < 1e-10 (~6.4 sigma) after ~1.5 months at ~200 m TVD, so a ~20 m, +0.6 g/cm3 body with 4 m2 of detectors is at that level. Same order of magnitude; their geometry is unknown to me. |

Written prediction: all six PASS. Actual: R2, R4, R6 PASS; R1, R3, R5 FAIL.

## What limits it

- **Heterogeneity, not photons, limits detection.** Z_emp barely grows with exposure. For the ore L = 20 cube at d = 120 it is
  5.9 / 8.3 / 6.6 / 11.3 / 9.5 at 30 / 45 / 90 / 180 / 365 d. The Asimov Z grows as sqrt(T): with the scale nuisance
  but no heterogeneity it is 9.1 / 11.2 / 15.8 / 22.3 / 31.8. The seed-to-seed spread of q under K0, driven by the 0.8 %
  smooth density field, sets a floor. That is why the carbonatite L = 20 cube at d = 120 stays below 5 even at 365 d
  (its Asimov Z at 180 d is 9.4).
- **Closer is easier.** "Depth below surface" runs opposite to distance from the drift. Bodies at d = 160 (40 m above the
  detectors) subtend more solid angle and are the easiest; bodies at d = 40 (160 m above) are the hardest.
- **Size threshold.** L = 5 m is never detected at any contrast. L = 10 m is detected only for ore grade within 80 m of the drift.
- **Low contrast (+0.05) never works**, as pre-registered.
- **Localisation is good (0-4 m median) for everything clearly detected** and breaks for marginal cells (L = 10). There
  the template scan can lock onto heterogeneity minima, some just above the drift. Cells that are not detected give
  random scan estimates; 107 of 264 of those land on the scan-box boundary.
- The MLEM reconstruction (figure only) shows the ore cube only after 10 m smoothing, smeared along the viewing
  directions (limited-angle geometry from one drift). Unsmoothed, the 2 m voxels are dominated by counting noise (~100 counts
  per 1-degree bin at 180 d).

## Flux-scale caveat (R1 failure) and what it implies

Our Reyna + CSDA model gives 2.67x the vertical intensity of Mei & Hime eq. (1) at 540 mwe, but only 1.24x their eq. (4) total
flux. Both fits are extrapolated below their 1 km.w.e. range and are not mutually consistent at this depth by our model, so
which is closer to reality is not settled here. If the true flux were 2.67x lower, the photon statistics of T days equal ours at
T/2.67 days: 180 d then corresponds to our ~67 d. Every ore-grade L = 20 cell is already detected at both 45 d and 90 d
(Z_emp 6.0-9.6, AUC 1.00), so R2 would not change. The Asimov (photon-limited) Z values would scale by ~0.61. This is a
scaling argument, not a separate run.

## Caveats

- Flat topography, sea-level spectrum, Reyna beyond its stated momentum range, CSDA with constant b.
- No scattering, detector resolution or efficiency; point detectors.
- Cube bodies rather than a tabular dipping ore zone.
- One heterogeneity model (1.5 % node s.d. on 30 m, my assumption).
- The detection statistic knows the body position; the template knows L and drho.
- Z_emp is a standardised separation over 8 seeds. It is not on the same scale as the Asimov Z and can exceed it
  (e.g. d 120 / L 10 / +0.60: Z_emp 7.70 vs Asimov-with-nuisance 5.7) because sd(q_K0) is estimated from 8 values.

## Files

Code:
- `src/check_ree_muo.cu`
- `tools/ree_muo_rules.py`
- `tools/render_ree_muo.py`

Outputs in `experiments/ree_muography/`:
- run_log.txt, flux_sanity.txt, validation.txt, heterogeneity.csv
- summary_detection.csv, q_values.csv, localisation.csv, cells.csv, detectors.csv
- blind_seed.txt, blind_estimate.txt, blind_truth.txt
- ree_muo_rules.json, meta.txt
- binary maps: rate_*, counts_*, scan_dF_*, recon2m/truth2m/domain2m, xb, acceptance, open_fraction
- figures/

Figures:
- fig_muo_scene_3d.png, fig_muo_orbit.mp4
- fig_muo_excess_maps.png, fig_muo_detectability.png, fig_muo_curves.png
- fig_muo_scan.png, fig_muo_recon_slices.png
