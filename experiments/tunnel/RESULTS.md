# ADR-012 results: tunnel NLOS through dust + ghost imaging vs camera

Frozen pre-registration: `experiments/tunnel/ADR-012.prereg-frozen.md`,
sha256 `99db809c434f3a0c748e3bf328f2a8453492f0eaf83b7966a29e58b4179a0276`, frozen 2026-09-28T04:38:07+02:00,
git HEAD 2747e94 (before any ADR-012 code existed). Everything below is synthetic.

## Runs

| run | command | wall |
|---|---|---|
| A, frozen grid (B = 1e10..1e13) | `build\check_tunnel.exe experiments\tunnel 20 8 256` | 38.1 s (render 21.0 s, recon 15.8 s) |
| A, exploratory x100 (B = 1e12..1e15) | `build\check_tunnel.exe experiments\tunnel\exploratory_budget_x100 20 8 256 100` | 39.3 s |
| A, exploratory x1e4 (B = 1e14..1e17) | `build\check_tunnel.exe experiments\tunnel\exploratory_budget_x1e4 20 8 256 10000` | 41.9 s |
| A statistics | `python tools\tunnel_stats.py experiments\tunnel` (+ same on both exploratory dirs with `--exploratory`) | < 5 s each |
| B, frozen | `python tools\ghost_tunnel.py experiments\tunnel\ghost 8` | 130 s |
| B, exploratory return-only | `python tools\ghost_tunnel.py experiments\tunnel\ghost\exploratory_return_only 8 --return-only` | 13 s |
| figures | `python tools\render_tunnel.py experiments\tunnel` | 20 s |

Build: `nvcc -O3 -arch=sm_120 -std=c++17 -lineinfo -diag-suppress 20013,221 -o build\check_tunnel.exe src\check_tunnel.cu`
(vcvars64). Seeds: part A seeds 1..8 (std::mt19937_64(0xADC012*1000003+seed) for relay texture/tilt and
target jitter; path hash seed 0x51ED270B ^ seed*0x01000193 shared by the three scenes of a seed, i.e.
common random numbers; Poisson streams curand Philox keyed by (seed, sensor, level, budget, pipeline,
scene)). Part B seeds 1..8, numpy default_rng([12, seed, budget_index, 10*tau]).
2^20 paths per relay point, 1024 relay points, 3 scenes per seed: 2.6e10 traced paths per run.

## Part A: frozen rules

Headline: **at the frozen budgets (B <= 1e13 first-bounce photons per relay point) the hidden person and
vehicle are not detected in ANY cell, even in clear air.** Max Z_emp over the whole frozen grid
(2 sensors x 7 visibilities x 4 budgets x 2 pipelines x 2 targets) = 2.56, max AUC = 0.906.

| rule | verdict | numbers |
|---|---|---|
| A1 clear-air baseline (V=inf, B=1e12, far, P2) | **FAIL** | person Z=1.41, AUC=0.734, localised 0/8 (median 3.91 m from box); vehicle Z=0.13, AUC=0.531, 0/8 (4.35 m) |
| A2 single scan P1 (same cell) | not usable (both) | person Z=0.41, AUC=0.625, 0/8; vehicle Z=0.80, AUC=0.688, 0/8 |
| A3 V_break(person, P2, far, 1e12) in {15, 20, 30} m | **FAIL** | V_break = none (fails already at V = inf) |
| A4 nothing detected at V = 5 m | **PASS** | 0 of 32 V=5 cells detected (also none in the exploratory runs up to B = 1e17) |
| A5 near sensor lowers V_break by >= 1 step | **FAIL** | far: none, near: none |

Why (photon accounting, expected per 1024-point scan at B = 1e12, seed medians): the tunnel itself returns
9.5e6 photons (clutter) and the person adds only 2.3e4 net photons (22 per relay point, spread over tens
of bins); vehicle 2.1e4. The clean (noise-free) change-detection volume does peak on the person's near face
(seed 1: peak at x 7.65, y 1.35, z 3.35, inside the true box) and on the vehicle's -x face, so geometry and
reconstruction are right; but the clean peak is only 0.18 sd of the B = 1e12 noise volume for the person
and 0.035 sd for the vehicle. Dust makes it worse fast:

| V (m) | tunnel+dust photons (far) | dust backscatter | person net, far | person net, near | vehicle net, far |
|---|---|---|---|---|---|
| inf | 9.46e6 | 0 | 2.31e4 | 2.31e4 | 2.07e4 |
| 50 | 3.26e6 | 2.09e6 | 1.9e3 | 5.1e3 | 1.5e3 |
| 30 | 1.36e6 | 1.06e6 | 356 | 1.85e3 | 256 |
| 20 | 4.19e5 | 3.62e5 | 44 | 517 | 28 |
| 15 | 1.21e5 | 1.10e5 | 5.4 | 144 | 3.1 |
| 10 | 9.2e3 | 8.8e3 | 0.08 | 11 | 0.04 |
| 5 | 3.0 | 2.9 | 3e-7 | 4e-3 | 7e-8 |

## Part A: where it starts working (EXPLORATORY, not pre-registered)

Same code, budgets multiplied by 100 and 1e4 (separate folders). Frozen statistic T:
- person, P2, far sensor: detected + localised at B = 1e16 (Z = 10.9, AUC 1.000, 8/8, median 0.02 m) and
  1e17 (Z = 36.3); first V > inf passing: V = 50 m needs B = 1e17 with the near sensor (Z = 9.2).
- vehicle: only at B = 1e17, near sensor, clear air (Z = 5.7); far sensor 1e17 gives Z = 4.4 (AUC 1.0,
  8/8 localised) -> fails Z >= 5.
- P1 (single scan, no reference): never detected, up to B = 1e17 (tunnel clutter dominates the max).
- Noise-whitened statistic (vol / propagated Poisson sd, also exploratory): same threshold for the person
  (1e16, Zw = 47.7) but reaches further into dust: person near-sensor V_break = 50 m at 1e16 and 20 m at
  1e17; vehicle detected from 1e16 (near) / 1e17 (far).
- Nothing at B <= 1e15 in any cell, frozen or whitened.
Scale: at the illustrative 2e13 detected first-bounce photons per second per watt (905 nm, 5 cm aperture,
11 m, rho = 1), B = 1e16 is ~500 W*s per relay point, i.e. ~5e5 J for the 1024-point scan.
Because both signal and clutter shot noise scale with A_p*rho_w^2, the threshold budget scales as
1/(A_p*rho_w^2): a 1 cm spot on rho ~ 0.2 rock is ~16x worse than rho ~ 0.8 white paint (model scaling,
not a separate run).

## Part B: frozen rules

Median PSNR (dB) over 8 seeds, N = 1e7 (primary):

| tau | camera | GI | camera+Wiener | GI+Wiener | camera SSIM | GI SSIM |
|---|---|---|---|---|---|---|
| 0 | 40.58 | 11.01 | 40.58 | 11.01 | 0.991 | 0.178 |
| 0.5 | 30.81 | 10.49 | 35.54 | 10.40 | 0.963 | 0.110 |
| 1 | 23.95 | 10.24 | 30.69 | 10.18 | 0.871 | 0.074 |
| 2 | 17.49 | 10.08 | 21.97 | 10.11 | 0.598 | 0.073 |
| 3 | 15.02 | 10.06 | 16.40 | 10.27 | 0.392 | 0.078 |
| 4 | 14.02 | 10.04 | 14.41 | 10.56 | 0.285 | 0.089 |
| 5 | 13.57 | 10.03 | 13.85 | 10.69 | 0.235 | 0.098 |
| 6 | 13.34 | 10.03 | 13.85 | 10.90 | 0.207 | 0.097 |

(~10.0 dB is the floor an affine fit of pure noise reaches for this target.)

| rule | verdict | numbers |
|---|---|---|
| B1 (N=1e7): "GI beats camera at OD >= X" | **no X**: "no OD in [0, 6] where GI beats the camera by >= 1 dB" | GI wins 0/8 seeds at every tau; median GI - camera = -29.5, -20.3, -13.7, -7.4, -5.0, -4.0, -3.5, -3.3 dB |
| B1-pred: no GI win at tau <= 2 | **PASS** | 0/8 at tau 0, 0.5, 1, 2 |
| B2 N=1e5 | no X | median GI - camera -8.7 ... -0.3 dB (both near floor at high tau) |
| B2 N=1e9 | no X | -35.8 ... -2.2 dB |
| B2 Wiener vs Wiener, N=1e7 | no X | -29.5 ... -2.9 dB |

Why: with equal photons and shot noise, each bucket reading carries the Poisson noise of the whole
half-illuminated scene, so each GI pixel's noise is ~sqrt(N_pix * mean(R) / R_i) ~ 64x (~36 dB) worse
than a camera pixel's; at tau = 0 that is exactly the observed ~30 dB gap. Dust narrows the gap only
because the camera also falls toward the noise floor. The frozen model is symmetric (dust on both paths),
which is the regime Bina et al. (PRL 2013) report as "equivalent to diffusive imaging", not better.

Exploratory (not pre-registered): dust ONLY between target and detector (Tajahuerce et al. 2014-like
geometry, clean illumination, no veil). At N = 1e9 GI beats the raw camera at every tau >= 2 (8/8 seeds,
+4.2 to +7.8 dB; X = 2), but Wiener-deconvolved camera still beats GI up to tau = 4 (25.7 vs 21.4 dB)
and loses only at tau >= 5 (17.9 vs 21.3; 15.0 vs 21.3). At N = 1e5 and 1e7 GI never wins.
So a single-pixel advantage appears in this model only with (i) scattering on the return path only,
(ii) a large photon budget, and (iii) against a camera without deconvolution, or at tau >= 5 with it.

## Caveats (declared in the ADR, restated)
- Single scattering; ballistic-only surface returns; no multiple scattering, no forward-scatter return;
  ideal time gate, no pile-up; flat relay geometry (roughness via albedo texture and normal tilt only).
- Near-sensor variant reuses the far transients and changes only the sensor-leg attenuation.
- Wording clarification (not a change): the frozen text states the repo per-path weight with rho_w once
  "relative to" the direct return; expressed in detected photons relative to B (rho = 1) the hidden and
  dust terms carry rho_w^2 (one factor per wall bounce), which is what the code does.
- Pulse applied by linear deposit + bin-integrated-Gaussian convolution instead of per-path splat_pulse
  (same integral, speed).
- Z_emp from 8 empty seeds has sizeable sampling scatter (e.g. near vs far in clear air differ only by
  noise yet give Z 16.6 vs 10.9 at 1e16).
- Part B is an analytic linear image-formation model, not transport; 64 x 64 pixels; fixed target.
- No claims about any company's system.

## Files
- Part A: tunnel_trials.csv (all 2688 volumes per run: T, peak, distance, whitened stat), tunnel_cells.csv,
  tunnel_rules.json, tunnel_photons.csv, tunnel_targets.csv, tunnel_meta.txt, tunnel_mips*.f32/.csv,
  tunnel_vol_*.f32, tunnel_wave_*.f32; same in exploratory_budget_x100/ and exploratory_budget_x1e4/.
- ghost/: ghost_results.csv, ghost_rules.json, ghost_images_seed1.npz (+ exploratory_return_only/).
- figures/: fig_tunnel_scene, fig_tunnel_volumes, fig_tunnel_detection_map_frozen, fig_tunnel_detection_map_whitened,
  fig_tunnel_curves, fig_tunnel_waveforms, fig_ghost_grid_frozen, fig_ghost_curves_frozen,
  fig_ghost_grid_return_only_exploratory, fig_ghost_curves_return_only_exploratory (.png).
