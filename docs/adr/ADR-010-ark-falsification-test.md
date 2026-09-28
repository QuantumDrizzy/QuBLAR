# ADR-010 -- Ark falsification test: can muography tell the Durupinar formation from a buried hull?

**Status:** Proposed -- PRE-REGISTRATION. Written and saved before any `check_ark` run.
Everything in sections 1-7 is frozen at the SHA-256 recorded in
`experiments/ark/PREREG_SHA256.txt`; results go in section 9 (added after the run) and
may not change sections 1-7.
**Date:** 2026-09-28
**Depends on:** ADR-006 (muon mode: marcher, sky sampling, direction binning, MLEM).

## 0. Why this is a falsification test, not a search

The claim under test ("the Durupinar formation is a buried ship") makes a density
prediction; muon transmission measures column density. The experiment asks a narrow,
answerable question on SYNTHETIC scenes: *with a stated, realistic detector set and
exposure, would a muography campaign separate the natural-formation hypothesis from each
hull hypothesis at >= 5 sigma?* A "yes" means a real campaign could falsify one side; a
"no" means muography cannot settle it at that exposure. Nothing here is real data, and no
result below says anything about what is actually in the ground.

## 1. Background (sources)

| Fact | Value used | Source |
|---|---|---|
| Location | Mt. Tendurek, beside Uzengili village, ~16 km SE of Dogubayazit, Agri Province; 39.4407 N, 44.2348 E | Wikipedia "Durupinar site" (en/tr) |
| Elevation | 1,966-2,004 m a.s.l. (Avci: ~1,900 m) | Wikipedia; Avci 2007 |
| Length | 157-164 m (164 m / 538 ft most quoted; 157 m = 300 royal cubits claim) | Wikipedia; Collins 2016 |
| Width | ~42-48 m at the widest | Collins 2016 / promoter sites (42 m); ancientcontent.com 2025 (48 m) |
| Relief | ~4 m high block ("150 m x 50 m x 4 m high ship-like feature") | Avci, M. (2007) Bull. Eng. Geol. Environ., "Noah's Ark: its relationship to the Telceker earthflow" |
| Geology consensus | Natural: limonite/magnetite concentrations in steeply dipping sedimentary layers on the limbs of a doubly plunging syncline (Collins & Fasold 1996, J. Geosci. Educ. 44(4):439, doi:10.5408/1089-9995-44.4.439); or a slumped Miocene limestone block carried by the Telceker earthflow (Avci 2007); MTA (Atabey, 2025) report: landslide-formed geological structure. Snelling (1992, Creation 14(4)) also natural. 1960 excavation found "only soil and rocks". | as cited |
| Past geophysics (claims) | 1985 GPR (Fasold/Wyatt, plus a "frequency generator" criticised as dowsing); 1987-88 GPR/coring (Baumgardner, Bayraktutan); 2014 ERT (Larson); 2019 ToPa 3D GPR 100/250 MHz: reflector at 1.6-1.9 m across the site, "rectangular" reflections ~7 m deep; promoters (Noah's Ark Scans, A. Jones) report corridor/room-like features to ~6 m, "three levels". None peer-reviewed as evidence of a hull. | noahsarkscans.com papers list; ToPa/"Integration of GPR, LiDAR..." (2023, symposium); Wikipedia |

Uncertainty on geometry is large (length +-4 m, width 42-50 m, relief 4 m but locally
higher at the western fault scarp; true depth of the body unknown). The model uses one
representative geometry (section 2) and says so.

## 2. Geometry (synthetic, frozen)

Coordinates: x along the long axis (bow +x), y across, z up; z = 0 is the surrounding
grade (FLAT -- the real site sits on a slope; not modelled).

- Simulation grid: 0.5 m voxels, x in [-104, 104], y in [-72, 72], z in [-22, 6] m
  (416 x 288 x 56). Margins chosen so every ray below grade stays inside the grid up to
  the 70 deg zenith cap (checked at run time: rays leaving the grid below z = 0 must be 0).
- Mound planform: length L = 160 m, max width W = 45 m; half-width
  w(x) = W/2 * sqrt(1 - u^2) at the stern (x < 0, rounded) and W/2 * (1 - u^2) at the bow
  (x > 0, pointed), u = x/(L/2). Surface height h = 4 m * sqrt(1 - (y/w)^2) (relief 4 m,
  Avci 2007).
- Formation body: the planform extruded from z = -16 m up to the mound surface.
- Ground outside the body (and below it): colluvium.
- Hull (H_B*): Genesis 300 x 50 x 30 cubits at the royal cubit: length 157 m, beam 26 m,
  height 15.7 m. Half-beam b(x) = 13 m * min(1, (78.5 - |x|)/15) (ends taper over 15 m).
  Top at z = +2.5 m (under ~1.5 m cover, where the 1.6-1.9 m GPR reflector is reported),
  bottom at z = -13.2 m; clipped to stay >= 1 m below the mound surface. Shell (sides,
  bottom, roof) 1.0 m thick; two internal decks (3 storeys, 5.23 m each) 0.5 m thick;
  transverse bulkheads every 13 m, 0.5 m thick; two longitudinal walls at y = +-2 m
  (a 4 m central corridor), 0.5 m thick. Everything else inside = compartments.

## 3. Hypotheses and materials (densities in g/cm^3)

| | Inside formation body (outside hull) | Hull walls/decks | Compartments |
|---|---|---|---|
| **H_A natural** | doubly plunging syncline: layers concave-up, layer surface zeta = z - 6 m*(y/22.5)^2 - 6 m*(x/80)^2; thicknesses ~ U(0.5, 2.0) m, densities ~ U(2.0, 2.7) per layer; 3% iid voxel noise; no voids | -- | -- |
| **H_B1 hull, voids** | same as H_A realisation | wood 0.75 | air 0.0012 |
| **H_B2 hull, filled** | same as H_A realisation | wood 0.75 | sediment 1.9 |
| **H_B3 petrified hull** | same as H_A realisation | rock 2.5 | sediment 1.9 |
| **V_L void sweep** | H_A realisation + one air cube of edge L in {1, 2, 3, 4, 6, 8} m centred at (20, 0, -5) m | -- | -- |

Colluvium/ground: 2.0 with 3% noise. Air above ground: 0.0012.
Per seed k the background realisation (layers + noise) is drawn with seed k; H_B*/V_L
realisation k = H_A realisation k with the hull/void inserted (paired backgrounds).
The analyst's MODELS are the ensemble means: formation 2.35 (mean of U(2.0,2.7)),
ground 2.0, no noise, with/without the hull or void.

## 4. Muon model and detector setup

- Opacity X = integral of rho ds along the ray (rho in g/cm^3, s in m, X in m.w.e.),
  marched with the repository's shared Amanatides-Woo `march_medium` (ADR-006 section 3)
  over the density grid.
- Survival: T(X, theta) = J(> p_min(X), theta) / J(> p0, theta), p0 = 0.2 GeV/c, with the
  Reyna (2006) / Bugaev sea-level spectrum I(p,theta) = cos^3(theta) I_V(p cos theta),
  I_V(z) = 0.00253 z^-(0.2455 + 1.288 lg z - 0.2555 lg^2 z + 0.0209 lg^3 z), and CSDA
  range with dE/dX = a + bE, a = 0.217 GeV/m.w.e., b = 4.0e-4 /m.w.e. (standard rock,
  PDG). This replaces ADR-006's single effective Beer-Lambert coefficient because the
  scenes here are shallow (40-200 m.w.e.) where the energy spectrum matters.
- Sky: the repository's cos^2(theta) sampler (`sample_sky`), zenith cap 70 deg, binned on
  `imaging_sky()` (1 x 1 deg, 360 az x 70 zenith bins).
- Flux normalisation: 1 muon cm^-2 min^-1 = 1.44e7 m^-2 day^-1 (PDG sea level, horizontal
  detector), spread over the bins by the cos^2 sampler. Site elevation (~2,000 m) raises
  the flux; NOT modelled, so sea-level exposure is conservative.
- Detectors: 6 stations x 1 m^2 (6 m^2 total), point-like with direction-independent
  acceptance, at z = -20 m (20 m below grade, ~24 m below the crest; requires boreholes /
  an adit under the formation), at x in {-45, 0, +45} m, y in {-15, +15} m.
- Expected maps: per scene and detector, 2^23 hash-keyed candidate rays (identical
  directions for every scene: ADR-006 section 5) give the per-bin mean transmission; the
  open-sky population per bin is assumed known (calibration run). Observed counts per bin
  ~ Poisson(N_b(E) * T_b) with N_b(E) = E_days * 1.44e7 * A * (open_b / 2^23).
- Exposures: E in {1, 7, 30, 90} days. Seeds: k = 1..8 (>= 3 required by the brief).
- Validation: one direct Bernoulli exposure (2^24 candidates, detector 0, H_A seed 1) must
  agree with the expected map: Pearson chi^2/ndf in [0.8, 1.2] over bins with >= 5 expected.

## 5. Statistic (frozen)

For a pair (H_A, H_X), X in {B1, B2, B3, V_L}:

- Models m^A_b(s), m^X_b(s): the ensemble-mean scenes, with a global density-scale
  nuisance s in [0.92, 1.08] (9 grid points, 2% apart; profile = minimum over the grid,
  refined by a parabola through the minimum and its neighbours).
- Poisson deviance D(d | m) = 2 sum_b [m_b - d_b + d_b ln(d_b/m_b)] over all bins of all 6
  detectors.
- Test statistic q = min_s D(d | A(s)) - min_s D(d | X(s))  (q > 0 favours H_X).
- For each seed k: q_A,k from data drawn under H_A realisation k, q_X,k from data drawn
  under H_X realisation k (independent Poisson draws).
- Empirical separation Z = (median_k q_X,k - mean_k q_A,k) / sd_k(q_A,k);
  AUC = fraction of (i, j) with q_X,i > q_A,j.
- Also reported (not decisive): Asimov Z_ideal = sqrt(D(m^X(1) | m^A(1))) (known
  background, no nuisance) and Z_nuis = sqrt(min_s D(m^X(1) | m^A(s))).

## 6. Decision rule (frozen)

> **H_A vs H_X is "distinguishable" if Z >= 5 AND AUC >= 0.99 over the 8 seeds at some
> exposure E <= 30 days with the 6 m^2 detector set of section 4. Otherwise
> "not distinguishable at <= 30 days" (and the 90-day result is reported as
> "would need longer").** The minimum detectable void is the smallest L in the V_L sweep
> that is distinguishable at <= 30 days. H_B3 is expected to be the hardest case; it is
> measured, not assumed.

A PASS here is a statement about the synthetic experiment's power, never a statement that
the formation is or is not a ship.

## 7. Declared limits before running

Synthetic geometry (one representative shape, flat terrain, formation depth assumed 16 m);
natural heterogeneity model is one choice (layered syncline + 3% noise) -- a natural
low-density body of hull shape would mimic H_B*; sea-level flux, no elevation, no
multiple scattering, no detector resolution/background/noise tracks, no open-sky
calibration error; ideal point-like detectors with 1 deg bins; 8 seeds give a coarse null
distribution (Z from a Gaussian approximation of 8 values).

## 8. Reproduce

```
nvcc -O3 -arch=sm_120 -std=c++17 -lineinfo -o build\check_ark.exe src\check_ark.cu
build\check_ark.exe experiments\ark
python tools\render_ark.py experiments\ark
```

## 9. Results

Added after the run (2026-09-28 03:56-03:57 CEST); sections 1-8 above are unchanged from the frozen
copy (`experiments/ark/ADR-010.prereg-frozen.md`, SHA-256 4C3E8CCF...CD605). Details, deviations and
limits: `experiments/ark/RESULTS.md`.

| pair | Z_emp at 1 / 7 / 30 / 90 d (8 seeds) | AUC | verdict (rule of section 6) |
|---|---|---|---|
| A vs B1 | 93.1 / 94.0 / 93.8 / 93.7 | 1.000 | distinguishable (1 d) |
| A vs B2 | 22.4 / 22.5 / 22.3 / 22.3 | 1.000 | distinguishable (1 d) |
| A vs B3 | 10.9 / 11.0 / 10.9 / 10.9 | 1.000 | distinguishable (1 d) |
| A vs B3x (post-hoc, walls only) | 2.5 / 2.7 / 2.6 / 2.6 | 0.98-1.00 | not distinguishable |
| min. detectable void | 2 m cube: 5.33 at 30 d (marginal; 4.80 at 90 d); 3 m: >= 5.46 at every exposure | | 2 m (marginal), 3 m robust |

Declared implementation deviation: the profile minimum uses golden-section on a cubic-Hermite interpolation
of ln m over the same 9-point s grid instead of a 3-point parabola (the parabola gave an impossible
Z_nuis = 0 for the 1 m void in a smoke run). Separation is heterogeneity-limited (flat in exposure), and it
comes from the bulk density of the hull volume, not from walls/decks (B3x). **Synthetic only: this measures
what a campaign could tell, not what is in the ground.**
