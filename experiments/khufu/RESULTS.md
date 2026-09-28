# ADR-011 Khufu muography validation: results (SYNTHETIC)

Everything here is a simulation made with QuBLAR's muon pipeline. None of it is real muon data. The published
numbers used for comparison are in ADR-011 section 1.

## Pre-registration
- Frozen file: `experiments/khufu/ADR-011.prereg-frozen.md`
- SHA-256 `8bbeebe9d8b7488715ed0b4fcc45138d8f1bb84660d959bdd2158291c77f5afd`
- Frozen at 2026-09-28T04:12:28+02:00, with git HEAD 2747e94 (see `PREREG_SHA256.txt`).
- The hash was re-checked after the run and is unchanged.

Timeline (all times Europe/Madrid):
- 04:12:28: ADR frozen.
- 04:16:33: `src/check_khufu.cu` final version written.
- 04:17:36: smoke test finished (2^20 rays, 3 seeds, gitignored `build/khufu_smoke`). The smoke test drew its own
  blind void, and that void is not part of any result.
- 04:17:53: full run started.
- 04:19:41: blind estimate written, followed by the blind truth.
- ~04:20: full run finished, 145 s wall time.

## Validation of the machinery
Bernoulli check on K0 (seed 1), detector NE1, with 2^24 rays: 154,855 expected and 154,996 detected,
chi2/ndf = 1.009 over 11,678 bins. **PASS.** The pre-registered window was [0.8, 1.2].

## Verdicts (ADR-011 section 6)

### (a) Detection: PASS for both voids
The table gives Z_emp over seeds 1-8, where the masonry heterogeneity changes with the seed and the density scale
is profiled over [0.92, 1.08].

| pair | f=0.1 | f=0.25 | f=0.5 | f=1 | f=2 | AUC (all f) |
|---|---|---|---|---|---|---|
| K0 vs K1 (6 Big-Void instruments) | 84.9 | 71.4 | 80.9 | 69.7 | 61.5 | 1.000 |
| K0 vs K2 (7 NFC instruments) | 50.5 | 50.2 | 51.1 | 50.3 | 50.0 | 1.000 |
| K0 vs K1i (reported only) | 87.7 | 97.2 | 77.8 | 63.7 | 58.0 | 1.000 |

- Both voids already pass Z >= 5 and AUC >= 0.99 at f = 0.1, i.e. one tenth of the published campaigns.
- Z_emp does not grow with exposure. The seed-to-seed spread of q under K0 (masonry heterogeneity) grows about as
  fast as the signal (∝ f), so Z_emp is limited by the heterogeneity, not by counting statistics.
- Asimov Z with the scale nuisance alone is much larger at f = 1: K1 242, K2 1367
  (`summary_detection.csv`).

### (b) Localisation: PASS
- **Big Void (K1):** 8/8 seeds within 5 m. The estimate is (7.2, 15.0, 60.0) in every seed, error 0.00 m.
- **NFC (K2):** 8/8 seeds within 1 m. The estimate is (6.95, 92.5, 21.0) in every seed, error 0.25 m.

Two things make these numbers look better than they are:
- The K1 true centre sits exactly on the 1 m scan grid, which is a favourable case.
- The carved NFC spans x = 6.0-8.0 on the 0.5 m voxel grid, so its voxel centre is x = 7.0. The 0.25 m "error"
  is therefore mostly this voxelisation offset.

Reported, not decisive: when the K1i (inclined) data are scanned with the horizontal template, the estimate is
(7.2, 16, 59), 1.41 m from the centre, in all 8 seeds.

### (c) Blind check: PASS
- The hidden centre came from an OS random seed. That seed is written to `blind_seed.txt` and was not printed.
- The scan estimate was written to `blind_estimate.txt` before the truth was written or printed. The program
  holds the truth in memory to build the scene, and the scan function takes only the data and the K0 model.
- Estimate: (13.2, -15.0, 67.0). Revealed truth: (13.00, -14.70, 66.48). **Error 0.63 m** (tolerance 5 m).
  The void was accepted on the first rejection-sampling try.
- The blind void spanned x 12.2-13.8, y -29.7 to 0.3 and z 62.2-70.8. That is 5.8 m east of the passage
  plane, about 30 m south of the K1 centre, and east of the King's Chamber granite block. It was found where it
  actually was, not at the known Big Void position.

### (d) Comparison with the papers: partly consistent, 18/25 checks
The comparison uses f = 1 (the published area × days). Z_reg = S/sqrt(B) over the anomaly region; tracks are
the expected K0 counts inside the acceptance.

| instrument | Z_sim | Z_paper | Z verdict | tracks sim | tracks paper | ratio | tracks verdict |
|---|---|---|---|---|---|---|---|
| NE1 (K1) | 40.0 | 13.7 | **NO (x2.9)** | 5.84e6 | 4.4e6 | 1.33 | yes |
| NE2 (K1) | 55.4 | 12.7 | **NO (x4.4)** | 8.54e6 | 6.2e6 | 1.38 | yes |
| H1 (K1) | 147.7 | >10 | yes | 2.17e7 | 4.8e6 | **4.51** | NO |
| H2 (K1) | 146.2 | >10 | yes | 5.96e7 | 12.9e6 | **4.62** | NO |
| G1+G2 (K1) | 4.13 | 5.8 | yes (x0.71) | G1 1.16e7 / G2 1.37e7 | 6.9e6 / 6.0e6 | 1.69 / **2.29** | yes / NO |
| EM1 (K2) | 9.98 | >10 | yes (>=5) | 3.09e8 | 9.48e7 | **3.26** | NO |
| EM2 (K2) | 720.8 | >10 | yes | 1.84e8 | 9.39e7 | 1.96 | yes |
| EM3 (K2) | 389.0 | >10 | yes | 4.69e7 | 2.90e7 | 1.62 | yes |
| EM4 (K2) | 180.4 | >10 | yes | 1.32e7 | 9.87e6 | 1.34 | yes |
| Charpak (K2) | 751.0 | >10 | yes | 1.42e8 | 7.34e7 | 1.93 | yes |
| Joliot (K2) | 662.0 | >10 | yes | 9.26e7 | 1.30e7 | **7.13** | NO |
| Degennes (K2) | 118.8 | >10 | yes | 3.00e7 | 2.98e7 | 1.01 | yes |

Significance checks pass 10 of 12. Both failures are the Nagoya emulsions in the Queen's Chamber, where the
simulation is 2.9-4.4 times more significant than the paper. Possible reasons, none of them tested:
- The papers computed significance differently (per-direction excess with systematics).
- Detector efficiency and resolution are not modelled.
- The real void may be smaller or shaped differently from the GG-section box.

Track counts pass 8 of 13. The simulation overshoots for H1/H2 (×4.5), EM1 (×3.3), Joliot (×7.1) and G2 (×2.3).
The published instruments disagree with each other in the same way. From the published numbers, tracks per
m² per day are:
- EM2: 1.98e6
- Charpak: 9.3e5
- Joliot: 3.7e5
- H1 (with "~5 months" taken as 150 d): 2.2e4

The simulation has no efficiency, quality cuts, flat-detector cos factor or real live time, so a factor-2 match
on raw counts is limited by what the model includes.

Also reported: G1+G2 for K1i gives Z_sim 6.10, closer to the published 5.8 than the horizontal K1 value (4.13).
This is not decisive and says nothing about which hypothesis is real.

### Direction check (reported only; `direction_check.csv`)
Median angle over seeds 1-8 between the pull>2 excess-weighted centroid and the direction to the true centre:

| instrument | median angle |
|---|---|
| H1 | 5.8 deg |
| H2 | 8.5 deg |
| NE2 | 10.9 deg |
| NE1 | 13.2 deg |
| G1 | 24.2 deg |
| G2 | 23.3 deg |
| EM3 | 1.7 deg |
| EM4 | 4.3 deg |
| Degennes | 13.3 deg |
| Charpak | 14.9 deg |
| Joliot | 26.4 deg |
| EM2 | 35.1 deg |
| EM1 | 52.0 deg |

The centroid is diluted by the hundreds to thousands of heterogeneity bins with pull > 2. For detectors very close
to the NFC, the void also fills a wide solid angle, so the centroid is not the centre direction. The template scan
(rule b) is the pre-registered localisation.

## Declared implementation choices and deviations
None of these change a pre-registered rule.
1. **Acceptance:** decided by the bin-centre direction. Rays in bins outside a detector's window are not marched.
   This saves time and does not change the result, because those bins never enter any statistic.
2. **Survival table:** extended to 800 mwe on 4,096 points (check_ark used 600 on 2,048), because chords through
   the pyramid base reach about 550 mwe with s = 1.08. The physics is the same.
3. **Geometry details left open by the ADR:**
   - z < 0 is bedrock at 2.2 g/cm³; no upward ray crosses it.
   - The antechamber is centred on x_p.
   - The al-Ma'mun tunnel's 2 m width is measured horizontally, perpendicular to its axis.
   - For the blind void, "known structure" means the air structures. The 5 m surface margin is perpendicular to the
     face and is also applied to the truncated top.
4. **MLEM figure:** uses kappa = 0.01 /mwe (0.05 in the ark run), because pyramid opacities are about 5× larger.
   It is a figure only.
5. **Smoke test:** 2^20 rays, 3 seeds, output under the gitignored `build/khufu_smoke`, with its own random blind
   void. Its (b) verdict read FAIL only because 3 seeds cannot meet a 7-of-8 rule. The full run's
   code was unchanged.
6. **Paper comparison:** G1+G2 has one Z check shared by both detectors, and each detector has its own track
   check. The 25 checks are 12 Z checks plus 13 track checks.

## Limits (from ADR section 7, plus what the run showed)
- The building geometry is approximate (±2 m) and the faces are smooth.
- The model has no air shafts, no casing and no extra density at the Chevron.
- Flux is Reyna with CSDA energy loss.
- Not modelled: scattering, resolution, efficiency and the flat-detector cos factor.
- Detectors are points.
- There is one heterogeneity model and 8 seeds.
- The simulation knows the true building (K0) exactly, apart from the masonry noise and a ±8% global scale. That
  is why the significances are so large. A real analysis has a much less certain K0.
- The published significances come from different statistics, so rule (d) is only an order-of-magnitude
  comparison.

## Figures (`experiments/khufu/figures/`)
- `scene3d_cutaway.png`: pyramid, known chambers, both Big Void hypotheses, NFC and detectors.
- `excess_maps_expected.png` and `excess_maps_seed1.png`: per-detector σ-per-cell excess maps in
  tanθx/tanθy (cone detectors in axis-relative degrees).
- `localisation_scans.png`: template-scan ΔF slices for K1, KB (blind) and K2.
- `recon_slices_K12.png`: MLEM density inside the search boxes.
- `significance_vs_exposure.png`
- `sim_vs_paper.png`
- `orbit_khufu.mp4` and `orbit_khufu.gif`

## Reproduce
```
nvcc -O3 -arch=sm_120 -std=c++17 -lineinfo -diag-suppress 20013 -o build\check_khufu.exe src\check_khufu.cu
build\check_khufu.exe experiments\khufu            # args: out [log2M=23] [seeds=8] [recon=1]
python tools\render_khufu.py experiments\khufu
```
On Windows, run nvcc inside the VS 2022 vcvars64 environment. The blind void is OS-random on every run, so a
rerun places a new blind void. Everything else is deterministic.
