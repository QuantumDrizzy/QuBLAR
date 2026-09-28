# ADR-012: Tunnel vision -- NLOS around a mine-drift corner, and ghost imaging through dust

- Status: pre-registered (frozen copy + SHA-256 in `experiments/tunnel/` before any run)
- Date: 2026-09-28
- Scope: synthetic only. Motivation (autonomous underground equipment) is context; nothing here
  says anything about any company's technology.

## 1. Question

A) Can QuBLAR's confocal transient-NLOS pipeline, using a low-albedo rock wall of a mine drift
as the relay surface, detect and localise a person or a small vehicle hidden around an L-bend,
and at what airborne-dust visibility and photon budget does it break?

B) Does computational ghost imaging (single-pixel, Walsh-Hadamard patterns, bucket detector)
image a line-of-sight target through dust better than a conventional camera given the SAME
emitted photon budget, and above what optical depth, if any?

## 2. Code paths (reuse, declared)

- Transport: a copy of `transient_trace` (src/transient.cuh) in `src/check_tunnel.cu`, same
  cosine-hemisphere sampling, same per-path weight rho_w*rho_q*cos_q*cos_p/r^2, same CUDA BVH
  `traverse_bvh` (trace.cuh), same `splat_pulse` time convention. Added: (i) the relay-spot
  factor A_p/pi (the repo kernel normalises the hidden return to a direct return of 1; here
  both are expressed in detected photons), (ii) dust extinction, (iii) single-scatter dust
  backscatter, (iv) one waveform per dust level in one pass. The OptiX path exists only in
  bench_trace and is NOT used; the transient path in the repo is the CUDA BVH.
- Reconstruction: `laplacian_filter` (src/nlos.hpp) semantics + a GPU port of `backproject`
  (same linear bin interpolation and time convention). LCT (src/lct.hpp) is NOT used: it needs
  a rectified planar relay grid whose lateral footprint contains the target, and here the
  targets sit 3-9 m laterally outside the 4 m relay patch (down the side drift). Declared, not
  hidden.
- Part B: `tools/ghost_tunnel.py` (numpy), new; the repo has no single-pixel code path.

## 3. Scene (part A)

Coordinates: x lateral, y up, z along the access drift. Metres.
- Access drift A: x in [-2.5, 2.5], y in [0, 5], z in [-10, 6]. End wall z = 6.
- Side drift B (the L-bend, to +x): x in [2.5, 22.5], y in [0, 5], z in [1, 6].
- All rock surfaces Lambertian, albedo 0.20 (the hidden-scene material).
- Relay patch on the end wall z = 6 (normal -z): x in [1, 5], y in [0.5, 4.5], 32 x 32 relay
  points (cell centres, 12.5 cm pitch). Rough dark rock: per relay point albedo
  rho_w ~ U(0.10, 0.30) and normal tilted by U(0, 5 deg) at random azimuth, drawn per seed.
  Paths leaving a tilted relay point toward the wall (dir.z > -0.01) are counted as lost.
- Sensor (far, primary): (-1.5, 1.6, -4); L1 = |p - s| in 10.4-12.3 m. Sensor (near,
  secondary): (0.0, 1.6, 3.0); L1 in 3.3-6.5 m. Both see the whole patch and neither sees into
  drift B beyond the corner. The near variant reuses the same transients: only the per-relay
  sensor-leg attenuation exp(-2 sigma_t L1) differs (after gating, the sensor leg adds only
  a time shift that the backprojection removes). Declared approximation.
- Targets (in drift B, per-seed jitter dx, dz ~ U(-0.5, 0.5)):
  - person, albedo 0.50, height 1.70: two legs 0.15 x 0.85 x 0.20, torso 0.45 x 0.60 x 0.30,
    head 0.20 x 0.25 x 0.20 (x, y, z), centred at (8 + dx, -, 3.5 + dz);
  - vehicle, albedo 0.30: body 2.5 x 1.0 x 1.5 on y in [0.3, 1.3] plus cab 1.0 x 0.5 x 1.5 on
    y in [1.3, 1.8] at its -x end, centred at (10 + dx, -, 3.5 + dz);
  - empty control: tunnel only.
- Sensor model: 1280 bins x 100 ps, t0 = 2*min(L1)/c - 1 ns, Gaussian pulse FWHM 500 ps;
  relay spot 1 cm diameter (A_p = 7.85e-5 m^2); 2^20 paths per relay point; hidden legs traced
  to 20 m. Direct wall return gated out (bins before the earliest hidden arrival are not
  simulated); sensor-leg dust backscatter arrives before the gate and is ignored (ideal gate,
  no detector pile-up or dead time). Declared.
- Reconstruction volume: x in [4, 14], y in [0, 5], z in [1, 6], 10 cm voxels (100 x 50 x 50);
  grid_to_pulse_ratio = 2.7 (< 4).

## 4. Dust model (part A)

- Visibility V in {inf, 50, 30, 20, 15, 10, 5} m; extinction sigma_t = 3.912 / V
  (Koschmieder, 2 % contrast threshold).
- Single-scattering albedo omega0 = 0.9, Henyey-Greenstein g = 0.7 (assumed, forward-peaked
  mineral dust); backscatter phase p(pi) = (1 - g) / (4 pi (1 + g)^2).
- Surface returns: weight x exp(-2 sigma_t (L1 + r)) (ballistic only; forward-scattered light
  that still reaches the target is dropped -- conservative).
- Dust backscatter on the hidden leg, single scatter: per path, per range step dr,
  rho_w * A_p * sigma_s * p(pi) * cos_p * exp(-2 sigma_t r) / r^2 * dr for r from 0.1 m to the
  surface hit, sigma_s = omega0 sigma_t, times exp(-2 sigma_t L1). Accumulated from all 2^20
  paths through a per-relay hit-distance histogram. Multiple scattering ignored. Declared.

## 5. Photons and noise (part A)

- Budget B = detected first-bounce photons per relay point for rho_w = 1 in clear air (the
  direct-return scale), B in {1e10, 1e11, 1e12, 1e13}. Illustration only: 1 W at 905 nm with
  a 5 cm aperture at 11 m gives ~2e13 detected first-bounce photons/s for rho = 1.
- Expected counts per bin = B x waveform + background 1e-14 x B per bin. Poisson draws (curand
  Philox). The total expected target photons per scan are reported per cell.
- Pipelines: P1 = single scan. P2 = change detection: scan minus an independent Poisson scan of
  the same seed's empty tunnel (same relay texture: the reference is "the corner before anyone
  walked in"). The empty control in P2 is empty minus an independent empty reference.

## 6. Statistics and pre-registered rules (part A)

- Filter: temporal Laplacian (nlos.hpp) of the counts; GPU backprojection.
- Test statistic T = maximum of the backprojected volume over interior voxels (centres at least
  0.3 m from floor, ceiling, walls z = 1 and z = 6, and from the x-faces of the volume).
- Seeds 1..8 (relay texture, target jitter, path seed, Poisson streams). Per cell
  (target, V, B, sensor, pipeline): Z_emp = (median T_target - mean T_empty) / sd(T_empty)
  over the 8 empty seeds; AUC = fraction of the 64 (target, empty) pairs with T_t > T_e.
- detected := Z_emp >= 5 AND AUC >= 0.99.
- localised := in >= 7 of 8 seeds the interior peak voxel lies within 0.5 m of the target's
  axis-aligned bounding box.
- V_break(target, B, sensor, pipeline) := the smallest V in the grid such that that cell and
  every cell with larger V are detected AND localised ("none" if V = inf fails).

Rules (verdicts PASS / FAIL, with numbers):
- **A1 (clear-air baseline).** V = inf, B = 1e12, far sensor, P2: person detected and localised
  AND vehicle detected and localised.
- **A2 (single scan).** Same cell with P1, per target. No directional prediction; P1 is called
  usable only where it passes.
- **A3 (break-point prediction).** For the person, P2, far sensor, B = 1e12:
  V_break in {15, 20, 30} m. PASS iff so.
- **A4 (hard floor prediction).** No (target, B, sensor, pipeline) combination is detected at
  V = 5 m. PASS iff none is.
- **A5 (standoff prediction).** Near sensor lowers V_break for the person (P2, B = 1e12) by at
  least one grid step relative to far. PASS iff so.
- Also reported (not rules): median localisation error per cell, detection map over B x V,
  expected target photons per cell.

## 7. Part B: ghost imaging vs camera through dust

- Target: fixed 64 x 64 reflectance image (person silhouette + bar chart), values in
  [0.05, 0.9]. Seeds 1..8 = independent noise realisations.
- One-way optical depth tau in {0, 0.5, 1, 2, 3, 4, 5, 6}.
- One-way transfer of an image or pattern I: T[I] = e^-tau I + (1 - e^-tau) omega0 eta_f
  (G_s * I), omega0 = 0.9, eta_f = 0.5, G_s Gaussian with s = 1.5 sqrt(tau) pixels (heuristic
  small-angle forward-scatter blur; 'nearest' edges so a flat field stays flat).
- Dust backscatter veil: v0 (1 - e^-2tau) times the emitted light, v0 = 0.05 (relative to a
  white clear-air return).
- Photon budget N = total photons a white target returns in clear air; equal for both systems;
  N in {1e5, 1e7 (primary), 1e9}.
- Camera: flood illumination, per pixel lambda = (N/4096) T[R . T[1]] + veil; Poisson plus
  2 e- rms Gaussian read noise per pixel (one frame).
- Ghost imaging: 4096 complementary pairs of Sylvester Walsh-Hadamard patterns (8192
  measurements); per-pixel illumination a = N/4096^2 per pair; bucket collects everything that
  returns: lambda_k = a eta_ret sum(R . T[m_k]) + veil_k, eta_ret = e^-tau + (1 - e^-tau)
  omega0 eta_f; Poisson plus 2 e- read noise per measurement; reconstruction
  x = H (b+ - b-).
- Secondary baselines: Wiener deconvolution with the known PSF of both images, K chosen
  oracle-best from {1e-4, 1e-3, 1e-2, 1e-1} per image (generous to both; declared).
- Scoring: each image rescaled by a least-squares affine fit to the truth, then PSNR (peak 1)
  and SSIM (11 px Gaussian window, sigma 1.5, data range 1).

Rules:
- **B1 (primary).** At N = 1e7, GI beats camera at tau iff GI PSNR >= camera PSNR + 1 dB in
  >= 7/8 seeds. The claim "GI beats camera at OD >= X" is made iff some X exists such that GI
  beats camera at every tau >= X in the grid; X is reported. Otherwise the reported claim is
  "no OD in [0, 6] where GI beats the camera by >= 1 dB".
- **B1-pred (prediction).** GI does NOT beat the camera at any tau <= 2 at N = 1e7. PASS iff so.
- **B2.** Same rule at N = 1e5 and 1e9, and for Wiener-deconvolved GI vs Wiener-deconvolved
  camera at N = 1e7. Reported, no prediction.
- Expectation written in advance: the model is nearly symmetric (GI's illumination-path blur
  mirrors the camera's return-path blur); a GI advantage can only come from the bucket using
  scattered return light that the camera sees as blur/veil, and read noise over 8192 vs 4096
  reads works against GI at low N. A null result is a valid outcome.

Sanity literature:
- Tajahuerce et al., Opt. Express 22(14):16945-16955 (2014): single-pixel imaging through
  dynamic scattering media between object and detector.
- Bina et al., PRL 110, 083901 (2013): backscattering differential ghost imaging in turbid
  media; contrast better than direct imaging but equivalent to diffusive imaging.

## 8. What is cut / declared up front

- Single scattering only; no multiple scattering, no forward-scatter return paths in part A.
- Ideal time gate, no SPAD pile-up/dead time, no ambient light beyond the flat background.
- Relay wall flat in geometry (roughness enters through albedo texture and normal tilt only).
- Part B is an analytic linear image-formation model, not a transport simulation.
- 8 seeds per cell (the minimum the brief allows).

## 9. Results

Filled 2026-09-28 after the runs (docs copy only; the frozen copy in experiments/tunnel/ is unchanged,
sha256 99db809c434f3a0c748e3bf328f2a8453492f0eaf83b7966a29e58b4179a0276). Details, commands and seeds:
experiments/tunnel/RESULTS.md.

Part A (8 seeds, 2^20 paths/relay, 2688 reconstructed volumes per run):
- A1 FAIL: V=inf, B=1e12, far, P2: person Z=1.41, AUC=0.734, localised 0/8; vehicle Z=0.13, AUC=0.531, 0/8.
- A2: P1 not usable (person Z=0.41, vehicle Z=0.80, 0/8 localised).
- A3 FAIL: V_break(person, P2, far, 1e12) = none.
- A4 PASS: no cell detected at V = 5 m.
- A5 FAIL: V_break none for both sensors.
- No cell of the frozen grid is detected (max Z = 2.56, max AUC = 0.906). Per scan at B=1e12 the tunnel
  returns 9.5e6 photons and the person adds 2.3e4 (vehicle 2.1e4); the noise-free change-detection
  volume peaks on the targets, so the failure is photon starvation, not geometry.
- Exploratory (not pre-registered): person detected+localised from B=1e16 in clear air (Z=10.9, 8/8),
  at V=50 m only with B=1e17 (near sensor); vehicle only at 1e17; P1 never. Noise-whitened statistic:
  same budget threshold, reaches V=20 m at 1e17 (near sensor).

Part B (8 seeds):
- B1: no OD in [0, 6] where GI beats the camera by >= 1 dB at N = 1e7 (GI 0/8 wins everywhere; median
  GI - camera from -29.5 dB at OD 0 to -3.3 dB at OD 6).
- B1-pred PASS.
- B2: no X at N = 1e5, 1e9, nor for Wiener vs Wiener at 1e7.
- Exploratory: dust on the return path only, N = 1e9: GI beats the raw camera for OD >= 2 (+4.2 to
  +7.8 dB) but a Wiener-deconvolved camera loses only at OD >= 5.

Clarification (wording, not a change): in detected photons relative to B (rho = 1), the hidden and dust
terms carry rho_w^2 (one factor per wall bounce); the code does this.
