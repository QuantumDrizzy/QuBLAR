# ADR-014 summary (REE carbonatite muography from a drift, synthetic)

- Frozen 2026-09-28T05:08:31+02:00, sha256 6d0f2b58...cc1a6ab2.
- Setup: four 1 m2 detectors in a drift 200 m down; 48 cube cells (L 5-40 m, depth 40-160 m, drho +0.05/+0.25/+0.60);
  8 heterogeneity seeds; 30-365 d exposures; 10.6 min GPU.
- R1 FAIL: the Bernoulli check passes (chi2/ndf 1.02), but the simulated vertical intensity is 2.67x Mei & Hime eq. (1)
  (extrapolated; the total flux is 1.24x their eq. (4)).
- R2 PASS: the ore-grade (+0.60) 20 m cube is detected at every depth within 30-45 d (Z_emp 6.4-11.3 at 180 d).
- R3 FAIL: the carbonatite (+0.25) needs L = 40 m except right above the drift (20 m at d = 160). The ore L_min is 10-20 m.
- R4 PASS: the low-contrast +0.05 body is never detected, even at 40 m.
- R5 FAIL (13/15): clearly detected cells are localised to 0-4 m (median); marginal 10 m ore cubes are not.
- R6 PASS: the blind 20 m ore cube is found within 0.93 m.
- Limit: in-situ density heterogeneity (0.8 % smooth field), not counting statistics. Z_emp is flat with exposure while
  the Asimov Z grows as sqrt(T).
