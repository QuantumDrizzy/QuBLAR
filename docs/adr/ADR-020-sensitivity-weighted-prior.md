# ADR-020 — A sensitivity-weighted prior: the fix for the sensor-ward bias

**Status:** DRAFT, not frozen. Once sections 1–6 are agreed they are frozen (SHA-256 in
`experiments/swp/PREREG_SHA256.txt`, plus a verbatim copy) before any engine code changes.
Results go in section 7 only.
**Date:** 2026-09-28
**Depends on:** ADR-017 (L1: prior as a plug-in), ADR-019 §5 (the U1 failure), `check_gravity`
(the depth-bias XFAIL).

## 1. What was looked at before this was written (disclosed, not held out)

`src/diag_u1_muons.cu` (commit alongside this draft) reproduced ADR-019's only U1 failure
(check_fusion, muons only, 2²³): 3 confident, 0 on the ore. It then measured it:

| test | result | reading |
|---|---|---|
| where | depth 31–33 m, on the vertical axis, 9–11 m above the x = 0 chamber, below the ore | on the ore's shadow cone, near its apex |
| who sees them (Fisher Σ w a², x=0 / x=−16 / x=+16) | 9.4–13.9 / 0.35–0.48 / 0.35–0.46; ore bits mean 2.43 / 0.67 / 0.67 | effectively one view, at 4–6× the ore's sensitivity |
| E(best branch) vs E(truth) | 31,853 vs 32,251 | **the model prefers the branches** by 398 nats |
| E(truth + false bits) − E(truth) | +98.5 | next to the truth they are not wanted: a substitute for the ore, not a feature of their own |
| other annealing seeds (3 × 16) | 2, 1, 1 off truth, always the same bits | not a sampler accident |
| other muon exposures (seeds 101, 201, 301) | 4, 0, 4 off truth, at 31 m | structural, not one noise draw |

**Mechanism.** One view cannot fix a body's position along its rays. Under an i.i.d. per-bit
prior (every bit costs κ ≈ 6.9 nats, every face λ = 2), the cheapest body that casts the same
shadow is the smallest one, and the smallest one sits where the ray bundle is narrowest: at the
sensor. Six bits near the detector cost about 90 prior nats. The 64-bit truth costs 634. The
posterior does what the prior says, and the prior is wrong about where bodies are.

Gravity shows the same shape: its undecided bits sit too shallow, next to the stations, and its
E(best) < E(truth) too. Li & Oldenburg name it for potential fields (1996, 1998): a sparse or
minimum-norm model concentrates where the sensitivity is highest. Their fix is to weight the
model cost by the sensitivity (Li & Oldenburg 2000: w_j = (Σ_i G_ij²)^{1/4}).

## 2. The fix (one rule, every sensor)

For each bit b, F_b = Σ over the assembled rows of w·a² (the diagonal of AᵀWA, all sensors
summed). F_ref is the median of F_b over bits with F_b > 0. The per-bit linear prior term
becomes

  κ_b = κ · max(1, √(F_b / F_ref))

λ is unchanged. The field is computed in `assemble()` from the rows alone. It never uses the
observed d, only geometry, exposure and σ, so it is fixed before the data are looked at.

- The exponent ½ is Li & Oldenburg's (w² = √Σ G²). It is not fitted.
- The floor at κ is one-sided: no bit is ever cheaper than the base prior. Bits the data barely
  see cannot become free, so the fix cannot open new places to invent a body.
- The median is the one declared choice with no literature value. It is declared here and not
  varied.

## 3. What it is not

It is not a Bayesian prior in the pure sense, because it depends on the sensor. It is a
correction for a known misspecification of the i.i.d. prior, stated as such. The evidence
budget's "prior nats" now uses κ_b, so every budget moves, and every check is re-run.

## 4. Test set

- **The target:** check_fusion muons-only, as in ADR-018.
- **Held out (never run before this freeze):**
  1. muon exposure seeds 401, 501 and 601, same scene;
  2. the ore moved to x = +8 m, centre depth 14 m, half-extent 3 m, at 2²³, 2²⁵ and 2²⁷;
  3. the check_fusion scene with gravity alone at σ = 1, and fused.
- **Every existing check:** check_ising, check_mine, check_ahead, check_gravity, check_fusion,
  plus the rest of `check.bat`.

## 5. Predictions and exit rules

- **P1 (must):** U1 holds in every run of §4: no confident bit off the truth.
- **P2 (must):** every control stays at 0 confident bits.
- **P3 (reported):** runs that paid and localised before still pay and localise (check_ising,
  check_mine at 2²⁷, the fused run). A loss is reported as the price of the fix, not hidden.
- **P4 (the question for gravity, reported):** check_gravity's σ = 1 run moves from XFAIL to a
  paid localisation, with depth error under 4 m.

If P1 or P2 fails anywhere, the fix is rejected, the engine keeps κ uniform, and ADR-019's
failure stands.

## 6. Alternative considered, not chosen

A claim-level refusal: no bit may be confident unless at least two views see it with a
comparable weight. It changes no posterior and no budget. Against it:
- it is muon-specific, since gravity has no "views";
- its threshold would be picked after seeing 0.03–0.05 on false bits against 0.28 on the ore;
- it hides the bias instead of removing it.

It stays the fallback if §5 rejects this fix.

## 7. Results

(after the freeze and the runs)
