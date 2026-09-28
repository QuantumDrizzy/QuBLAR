# ADR-011 -- Khufu validation: can QuBLAR's muon pipeline reproduce the Big Void and the North Face Corridor?

**Status:** Accepted (validation; results in section 9) -- PRE-REGISTRATION. Sections 1-8 are frozen before any `check_khufu` run at the SHA-256
recorded in `experiments/khufu/PREREG_SHA256.txt` (unmodified copy: `experiments/khufu/ADR-011.prereg-frozen.md`).
Results go in section 9 only.
**Date:** 2026-09-28
**Depends on:** ADR-006 (marcher, sky, binning, MLEM), ADR-010 (density-grid opacity, Reyna/CSDA survival table,
expected-map + Poisson counting, Bernoulli validation, profile-likelihood statistic). The muon machinery is
copied from `src/check_ark.cu` into `src/check_khufu.cu`; check_ark is not modified.

## 0. What this is

A validation, not a discovery: the voids are known. The questions are (a) does the pipeline detect them with
detectors and exposures like the published ones, (b) does it put them in the right place, (c) does it find a
void where it actually is when the analyst does not know where that is (blind), and (d) are the simulated
significances and counts of the same order as the papers'. Synthetic scenes only.

## 1. Sources and what they report (verbatim numbers only; "n/a" where not in an accessible source)

**Morishima et al., Nature 552, 386-390 (2017)** -- read via the authors' pre-submission manuscript
arXiv:1711.01576 (the final Nature text is paywalled; numbers could differ in the final version).
- Pyramid "139 m high and 230 m wide"; Grand Gallery "8.6 m high x 46.7 m long x 2.1 to 1.0 m wide".
- Big Void: "cross section similar to the Grand Gallery and a length of 30 m minimum", "above the Grand Gallery";
  "The centre of the void is located between 40 m and 50 m from the floor of the Queen's chamber"; "it could be
  inclined or horizontal"; KEK: "starts almost at the centre of the Pyramid and ends at an angle whose tangent
  is 0.8 to the North".
- Nagoya emulsions: NE1 in the "Niche" east of the Queen's Chamber (QC), NE2 near its south-west corner; NE1
  "5.8 m east from the axis of the Grand Gallery", NE2 "4.5 m west", 1.1 m apart N-S; effective area 0.45 m2;
  4.4 M tracks in 98 days (NE1), 6.2 M in 140 days (NE2); analysis acceptance |tan theta| <= 1.0; significance
  "13.7 sigma (statistical) for position NE1 and 12.7 sigma for position NE2" (">10 sigma at the highest
  difference direction"). Stone 2.2 g/cm3, granite 2.75 around the King's Chamber (KC); rock thickness 65-115 m.
- KEK hodoscopes: 1.2 x 1.2 m2; H1 near the SE corner of the QC (Aug 2016-Jan 2017, 4.8 M events, tan
  acceptance to 0.8); H2 2.9 m west (from Jan 2017, "more than eight months", 12.9 M events, to 1.2).
  Significance per slice ">10 sigma except for the most outer slice" (H1 >5 sigma, H2 >7 sigma there). No single
  combined number.
- CEA Micromegas: two telescopes of 50 x 50 cm2 "in front of the chevrons (North face), at a distance of 17 and
  23 m", axes slightly toward the east; May 4-July 3 2017; 6.9 M and 6.0 M good tracks; Big Void combined
  "5.8 sigma", Grand Gallery "8.4 sigma". Exact coordinates and tilt: n/a in text (figure only).

**Procureur et al., Nature Communications 14, 1144 (2023)** (open access).
- NFC: Nagoya W 2.02 +- 0.06, H 2.18 +- 0.17, L 9.06 +- 0.07 m; bottom "~20 m from the ground level"; north end
  0.84 +- 0.05 m behind the Chevron north face; E-W centred on the Chevron / above the descending corridor (DC);
  MC slope -0.3 +- 1.5 deg. CEA: L 9.23 +- 0.48 m, W 1.87 +- 0.11, H 1.85 +- 0.12 m, slope -1.9 (+7.2/-4.7) deg.
- Corridors run N-S "at a distance of about 7 m east from the center of the pyramid".
- Nagoya DC films EM1-EM4 (0.225 m2 per 3-film detector; the paper prints "cm2"), 172 / 211 / 211 / 79 days;
  9.48e7 (EM1), 9.39e7 (EM2N), 2.90e7 (EM3), 9.87e6 (EM4) tracks within |tan theta| <= 1; significance "well over
  10 sigma". CEA in the DC: Joliot 50 x 50 cm2, Degennes and Charpak 100 x 50 cm2; ~140 d (Joliot, Degennes),
  ~158 d (Charpak); 13.0 M, 29.8 M, 73.4 M good muons; significance "largely above 10 sigma". Best-fit stone
  density 2.2 g/cm3 (scan 1.8-2.6). Exact detector coordinates: n/a in text.

**Geometry (secondary):** base ~230.3 m, height now ~138.5 m (original ~146.6 m); entrance ~17 m up the north
face, ~7.2 m east of the axis; DC ~26.5 deg; QC ~5.75 x 5.23 m; KC 10.47 x 5.23 x 5.8 m, floor ~43 m; five
relieving chambers above the KC. Positions below are approximations (+-2 m) from these standard figures; they
are the SIMULATION truth, so the validation is self-consistent even where they differ from the real building.

## 2. Scenes (frozen)

Coordinates: origin at the base centre, x east, y north, z up. Passage plane x_p = +7.2 m.
Grid 0.5 m, x, y in [-120, 120], z in [-2, 140] (480 x 480 x 284). Pyramid: |x|,|y| <= 115.15 (1 - z/146.6),
0 <= z <= 138.5 (truncated top). Mean stone 2.2 g/cm3; granite 2.75 in the KC block
(x in [-4.3, 9.2], y in [-9.6, -1.4], z in [41.5, 65]).
Known structures (air), all in K0:
DC 1.05 wide, 1.34 m vertical, floor z = 17 - (101.8 - y) tan 26.5 deg from the entrance (y = 101.8) down to z = -2;
ascending corridor 1.05 x 1.34 from the DC junction (y = 76.7) to the Grand Gallery (GG) foot (y = 42.5, z = 21.7);
horizontal passage 1.05 x 1.2 at z = 21.7 from y = 42.5 to the QC; QC x in [2.0, 7.75], y in [-2.6, 2.6],
z in [21.7, 27.2]; Niche tunnel x in [7.75, 13.5], y in [-0.2, 1.3], z in [21.7, 23.7]; GG from y = 42.5 to
y = 0.6, floor z = 21.7 + (42.5 - y) tan 26.3 deg, 8.6 m high, width 2.1 m at the floor to 1.0 m at the roof;
antechamber y in [-2.4, 0.6], z in [43, 46.8], 1.5 m wide; KC x in [-2.77, 7.7], y in [-8.1, -2.9], z in [43, 48.8];
relieving chambers 1.0 m high over the KC footprint at z = 50, 53, 56, 59, 62; al-Ma'mun tunnel 2 x 2 m from
(0, 109.6, 7) to (x_p, 76.7, 5.5). Air shafts and the subterranean chamber are omitted (below every detector /
too small to see, as the 2017 paper notes).

- **K0** known structures only.
- **K1** K0 + Big Void, HORIZONTAL hypothesis: x in [x_p - 0.8, x_p + 0.8], y in [0, 30], z in [55.7, 64.3]
  (1.6 m x 8.6 m section ~ GG, 30 m long, centre (7.2, 15, 60): 41 m from the QC floor, i.e. inside the
  published 40-50 m).
- **K1i** K0 + Big Void, INCLINED hypothesis: same centre and size, long axis parallel to the GG (rising to the
  south at 26.3 deg). Reported, not decisive.
- **K2** K0 + NFC: x in [x_p - 1, x_p + 1], y in [88.0, 97.0], z in [20.0, 22.0] (2 x 2 x 9 m, north end 0.86 m
  behind the face at z = 22).
- **K12** K0 + K1 + K2.
- **KB (blind)** K0 + a Big-Void-sized horizontal N-S void at a hidden random centre drawn uniformly in
  x in [x_p - 12, x_p + 12], y in [-20, 40], z in [45, 80], rejected if within 10 m of the K1 centre, overlapping a
  known structure (2 m margin), or closer than 5 m to the pyramid surface. The seed comes from the OS at run time,
  is written to disk, and is not printed; the truth is revealed only after the localisation is written.

## 3. Masonry heterogeneity (nuisance), per seed

Courses: thickness ~ U(0.6, 1.5) m, per-course relative density offset N(0, 0.02); within a course, blocks of
U(1, 2) x U(1, 2) m with random offsets and relative density N(0, 0.05); plus 3% iid voxel noise. Applied to
limestone and granite. The analyst's models are the means (2.2 / 2.75, no noise) with a global density-scale
nuisance s in [0.92, 1.08] (as ADR-010). Seeds 1-8 (paired backgrounds per seed as ADR-010); blind background
seed 101.

## 4. Detectors (approximating the published placements; direction-independent acceptance inside the stated window)

Big-Void set: NE1 (13.0, 0.55, 22.0), NE2 (2.7, -0.55, 22.0): 0.45 m2 each, 98 / 140 d, |tan theta_x|,|tan theta_y| <= 1.
H1 (7.2, -2.0, 22.3): 1.44 m2, 150 d, |tan| <= 0.8; H2 (4.3, -2.0, 22.3): 1.44 m2, 250 d, |tan| <= 1.2.
G1 (8.7, 112.0, 5.0), G2 (9.7, 117.0, 0.5) outside the north face (17.4 and 24.1 m from the Chevron point
(7.2, 101, 18.5)): 0.25 m2 each, 60 d, cone of half-angle 40 deg around the direction to (7.2, 18, 47.5).
NFC set (in the DC, floor + 0.4 m): EM1 y = 100.5 (172 d), EM2 y = 97.5 (211 d), EM3 y = 88.0 (211 d),
EM4 y = 85.0 (79 d), 0.225 m2 each, |tan| <= 1; Charpak y = 93.0 (0.5 m2, 158 d), Joliot y = 95.5 (0.25 m2, 140 d),
Degennes on the DC/AC junction y = 77.5 (0.5 m2, 140 d), |tan| <= 1.
Muon model exactly as ADR-010 section 4 (Reyna 2006 + CSDA a = 0.217 GeV/mwe, b = 4e-4 /mwe, p0 = 0.2 GeV/c;
cos^2 sky, 70 deg zenith cap, 1 x 1 deg bins; 1 muon cm^-2 min^-1 sea level; Giza ~60 m altitude, ignored);
2^23 hash-keyed rays per detector per scene; one Bernoulli cross-check (chi2/ndf in [0.8, 1.2]).
Exposure sweep: factor f in {0.1, 0.25, 0.5, 1, 2} times each detector's published days.

## 5. Statistics (frozen)

- Detection: ADR-010 section 5 statistic over the bins inside each detector's acceptance, with per-detector
  exposure weights: q = min_s D(d | K0(s)) - min_s D(d | X(s)); Z_emp = (median q_X - mean q_K0)/sd(q_K0), AUC,
  over seeds 1-8.
- Localisation (template scan): candidate void of the hypothesis' shape (BV: 1.6 x 30 x 8.6 m horizontal N-S;
  NFC: 2 x 9 x 2 m horizontal N-S) on a grid (BV: x in [x_p - 20, x_p + 20], y in [-30, 50], z in [40, 80], step 1 m;
  NFC: x in [x_p - 5, x_p + 5], y in [80, 104], z in [16, 28], step 0.25 / 0.5 / 0.25 m). Expected counts
  N_b(c) = N_b(K0) * T(X_b - 2.2 L_b(c)) / T(X_b), X_b the K0 opacity of the bin-centre ray, L_b(c) its chord
  through the candidate; the estimate minimises sum_b [N_b(c) - d_b ln N_b(c)] over the set's bins. Uses only
  data and the K0 model -- never the truth.
- Direction (reported, not decisive): per detector, the excess-weighted centroid direction of bins with pull
  (d - N(K0))/sqrt(N(K0)) > 2, compared with the direction to the true centre.
- Comparison with the papers: Z_reg = S/sqrt(B) with S, B the expected excess and K0 counts summed over the
  "anomaly region" (bins whose expected excess >= 20% of the detector's maximum), at published area/exposure;
  CEA G1+G2 summed (the paper combined them). Track counts = expected K0 counts in the acceptance.

## 6. Decision rules (frozen)

> **(a) Detection:** K1 vs K0 with the six Big-Void instruments, and K2 vs K0 with the seven NFC instruments, are
> each "detected" if Z_emp >= 5 AND AUC >= 0.99 over seeds 1-8 at some f <= 1 (the published exposures).
>
> **(b) Localisation:** the template-scan centre is within 5.0 m (Big Void, K1) / 1.0 m (NFC, K2) of the true
> centre in at least 7 of the 8 seeds at f = 1.
>
> **(c) Blind check:** the KB void is localised by the same scan within 5.0 m of its hidden centre at f = 1, with
> the estimate written to disk before the truth is revealed.
>
> **(d) Comparison:** for each published instrument, Z_reg is "consistent" with the paper if
> 0.5 <= Z_sim / Z_paper <= 2 (for reported lower bounds "> 10 sigma": consistent if Z_sim >= 5); simulated
> track counts are consistent with published counts within the same factor of 2. Otherwise the discrepancy is
> reported with its size.

## 7. Declared limits before running

Approximate building geometry (+-2 m), smooth faces (no steps, no casing remnants, no Chevron density excess),
no air shafts; one heterogeneity model; Reyna flux (the 2023 paper preferred Guan), CSDA energy loss, no
multiple scattering, detector resolution, efficiency, noise or flat-detector cos factor; point detectors with
box/cone acceptance; published significances use different statistics (peak slices, Gaussian integrals), so (d)
is an order-of-magnitude comparison; 8 seeds.

## 8. Reproduce

```
nvcc -O3 -arch=sm_120 -std=c++17 -lineinfo -diag-suppress 20013 -o build\check_khufu.exe src\check_khufu.cu
build\check_khufu.exe experiments\khufu
python tools\render_khufu.py experiments\khufu
```

## 9. Results

Filled in after the run on 2026-09-28. Sections 1-8 above are byte-identical to the frozen copy (SHA-256
8bbeebe9d8b7488715ed0b4fcc45138d8f1bb84660d959bdd2158291c77f5afd, frozen 04:12:28+02:00; full run started
04:17:53+02:00, wall time 145 s). Details, tables and deviations are in `experiments/khufu/RESULTS.md`.

- Validation (Bernoulli, K0 seed 1, NE1, 2^24 rays): chi2/ndf = 1.009 (ndf 11678). PASS.
- **(a) Detection: PASS.**
  - K0 vs K1: Z_emp 84.9 at f = 0.1 and 69.7 at f = 1, AUC 1.000.
  - K0 vs K2: Z_emp 50.5 at f = 0.1 and 50.3 at f = 1, AUC 1.000.
  - Reported only: K1i gives Z_emp 63.7 at f = 1.
  - Z_emp is capped by the seed-to-seed masonry variance and does not rise with f.
- **(b) Localisation: PASS.**
  - K1: 8/8 seeds within 5 m, error 0.00 m. The truth lies on the scan grid.
  - K2: 8/8 seeds within 1 m, error 0.25 m, which equals the 0.5 m voxelisation offset of the carved corridor.
- **(c) Blind: PASS.** The OS-random hidden centre was (13.00, -14.70, 66.48). The estimate (13.2, -15.0, 67.0) was
  written before the truth. Error 0.63 m.
- **(d) Comparison: partly consistent, 18 of 25 checks.**
  - Significance, 10/12 consistent. NE1 and NE2 are too high in the simulation: 40.0 vs 13.7 (×2.9) and 55.4 vs
    12.7 (×4.4). G1+G2 gives 4.13 vs 5.8 (×0.71).
  - Tracks, 8/13 consistent. The simulation is high for H1 ×4.51, H2 ×4.62, G2 ×2.29, EM1 ×3.26 and Joliot ×7.13.
- **Direction check (reported only):** median angle between 1.7 deg (EM3) and 52 deg (EM1).
- **Status:** Accepted as a validation of the pipeline's detection, localisation and blind-placement behaviour on
  synthetic Khufu scenes. It is not a statement about the real building. The significance scale comparison with
  the papers is only order-of-magnitude, as pre-declared.
