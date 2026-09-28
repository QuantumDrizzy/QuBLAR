# ADR-010 results -- Ark falsification test (SYNTHETIC; no real data)

Run: 2026-09-28 03:56:10 -> 03:57:44 (Europe/Madrid, UTC+2), 94.2 s wall, RTX 5060 Ti (sm_120), repo HEAD 2747e94.
Pre-registration: `docs/adr/ADR-010-ark-falsification-test.md`, sections 1-8 frozen at 03:51:06 with
SHA-256 4C3E8CCF3A730AA66582E3E1C495C3FFC5E8F039E85C56AAF79AF3D9491CD605
(`PREREG_SHA256.txt`; unmodified copy: `ADR-010.prereg-frozen.md`). Full console output: `run_log.txt`.

## Commands

```
:: from a VS 2022 x64 prompt (or after vcvars64.bat), repo root
nvcc -O3 -arch=sm_120 -std=c++17 -lineinfo -diag-suppress 20013 -o build\check_ark.exe src\check_ark.cu
build\check_ark.exe experiments\ark 23 8 1      :: args: out dir, log2(rays/detector/scene), seeds, recon on/off
python tools\render_ark.py experiments\ark
```

Seeds: background realisations (layer thicknesses/densities + 3% voxel noise) = seeds 1..8;
Poisson draws `mt19937_64((0x5EED*1000003 + seed*1009 + e*17)*31 + hypothesis)` (e = exposure index);
Bernoulli validation seed 777; MLEM data Poisson seed 0xBEEF + hypothesis. 2^23 hash-keyed rays per
detector per scene. Deterministic: re-running the command reproduces every number below.

## Sanity / validation

| check | result |
|---|---|
| Reyna vertical J(>1 GeV) | 70.2 m^-2 s^-1 sr^-1 (PDG ~70) |
| vertical survival T(X) | 10 mwe 0.493, 40 mwe 0.119, 60 mwe 0.066, 100 mwe 0.0286, 200 mwe 0.00787 |
| rays leaving grid below grade | 0 |
| direct Bernoulli vs expected map (H_A s1, D0, 2^24) | expected 1,584,679, detected 1,585,072, chi2/ndf 0.916 (ndf 24,775): PASS |

## Decision (ADR-010 section 6: Z >= 5 AND AUC >= 0.99 over 8 seeds at <= 30 d, 6 x 1 m^2)

| pair | 1 d | 7 d | 30 d | 90 d | AUC (all E) | Asimov Z_nuis 1 d / 30 d | verdict |
|---|---|---|---|---|---|---|---|
| A vs B1 (hull, air compartments) | 93.13 | 94.02 | 93.79 | 93.67 | 1.000 | 2601 / 14248 | **DISTINGUISHABLE** (already at 1 d) |
| A vs B2 (hull, sediment-filled) | 22.42 | 22.51 | 22.29 | 22.34 | 1.000 | 478 / 2620 | **DISTINGUISHABLE** (1 d) |
| A vs B3 (petrified walls 2.5, fill 1.9) | 10.87 | 11.00 | 10.86 | 10.88 | 1.000 | 181 / 993 | **DISTINGUISHABLE** (1 d) |
| A vs B3x (POST-HOC: petrified walls only, fill = natural) | 2.53 | 2.66 | 2.61 | 2.61 | 0.984-1.000 | 30 / 162 | not distinguishable, not even at 90 d |

Void sweep (air cube of edge L at (20, 0, -5) m inside H_A), Z_emp:

| L | 1 d | 7 d | 30 d | 90 d | verdict (<= 30 d) |
|---|---|---|---|---|---|
| 1 m | 1.63 | 0.79 | 2.80 | 2.41 | not detectable |
| 2 m | 2.32 | 3.32 | **5.33** | 4.80 | detectable by the rule, MARGINAL (7 d and 90 d are below 5) |
| 3 m | 5.46 | 7.15 | 8.04 | 7.43 | detectable (robust: >= 5 at every exposure) |
| 4 m | 7.52 | 9.88 | 11.15 | 10.24 | detectable |
| 6 m | 16.61 | 16.70 | 18.27 | 17.21 | detectable |
| 8 m | 27.15 | 25.56 | 27.85 | 26.60 | detectable |

**Minimum detectable void: 2 m cube under the letter of the rule (Z = 5.33 at 30 d), but marginal; 3 m is the
smallest size that clears Z >= 5 at every exposure.** Full table: `summary.csv`; per-seed q: `q_values.csv`.

## What the numbers mean

- Z_emp does not grow with exposure: past ~1 day of 6 m^2 the test is limited by the natural heterogeneity
  (the analyst does not know the layering), not by counting statistics. The known-background Asimov Z grows as
  sqrt(E) (e.g. B3: 219 at 1 d -> 2074 at 90 d) -- that is the gap between an ideal and a realistic analysis.
- B1/B2/B3 are separable because they change the bulk density inside a 157 x 26 x 15.7 m volume
  (B3's sediment fill at 1.9 against a formation mean of 2.35), which the difference maps show as a broad
  excess of muons over the whole hull, not as a picture of walls and decks. B3x, where only the walls and decks
  change, sits at Z ~ 2.6 at every exposure: thin walls are invisible to this setup. The brief's expectation
  "petrified hull indistinguishable" holds for B3x, not for B3 as pre-registered (the fill decides).
- Consequently a natural low-density body of hull-like shape would look like B2/B3; muography constrains
  density distribution, not origin.

## Deviations and post-hoc additions (declared)

1. Profile minimiser: pre-registered "parabola through the minimum and its neighbours" replaced by golden-section
   search on a cubic-Hermite interpolation of ln m between the same 9 grid points (same s range). Reason: a smoke
   run (2^18 rays, 3 seeds, gitignored build/ark_smoke) gave Asimov Z_nuis = 0.0 for the 1 m void while Z_ideal
   = 3.6, which is impossible; the parabola error exceeded small signals. Changed before the full run.
2. B3x is POST-HOC (added after the smoke run showed B3 separable through its fill). It is labelled in every
   output and is not part of the pre-registered decision.
3. The pre-registered hull (26 m beam, 15 m taper) is wider than the pointed bow of the pre-registered mound
   planform for x ~ 52-78 m, so there it sits in colluvium below grade. Kept as registered.
4. Figures display the 1 m re-sampling of truth; 0.5 m decks/bulkheads can be missed in the 1 m display (the
   simulation itself runs at 0.5 m). The MLEM slices are illustrations and are not part of the decision.
5. Two smoke runs (2^18 rays, 3 seeds) preceded the full run; their numbers are not reported as results.

## Limits

Synthetic geometry (one shape, flat terrain, 16 m deep body, hull dimensions from Genesis at the royal cubit);
one natural-heterogeneity model; sea-level Reyna flux, no altitude gain (~2,000 m, conservative), CSDA energy
loss with constant a, b, no multiple scattering, no detector resolution, efficiency, background or open-sky
calibration error; point-like 1 m^2 detectors with direction-independent acceptance and 1 deg bins; detectors
20 m below grade under the formation (needs boreholes/adit -- a practical, not a physics, barrier); 8 seeds give
a coarse null distribution (Gaussian Z of 8 values; AUC resolution 1/64). No real data were used or produced.
