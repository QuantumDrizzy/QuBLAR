# ADR-012 summary (tunnel NLOS + ghost imaging through dust)

Frozen: sha256 99db809c...a0276 at 2026-09-28T04:38:07+02:00. Synthetic only.

- NLOS around the L-bend with a dark rock relay wall (rho 0.1-0.3, 1 cm spot, 11 m standoff) does NOT
  work at the pre-registered photon budgets (<= 1e13 first-bounce photons per relay point): no detection
  in any cell, clear air included. A1, A3, A5 FAIL; A2 P1 unusable; A4 (nothing at V = 5 m) PASS.
- Exploratory: change detection finds and localises the person from B = 1e16 (clear air; Z = 10.9,
  8/8 within 0.5 m); in dust it needs 1e17 and V >= 50 m (frozen statistic). The vehicle needs 1e17.
  Single scans (no empty reference) never work: tunnel clutter dominates.
- Ghost imaging vs camera with equal photons through dust on both paths: the camera wins at every
  optical depth 0-6 and every budget (1e5, 1e7, 1e9); at N = 1e7 by 29.5 dB (OD 0) to 3.3 dB (OD 6).
  B1: no OD where GI beats the camera; B1-pred PASS. Exploratory: with dust only on the return path and
  N = 1e9, GI beats the raw camera for OD >= 2, but a Wiener-deconvolved camera only loses at OD >= 5.
- Reproduce: see RESULTS.md. Figures: figures/*.png (copied to /workspace/qublar/mining/tunnel/).
