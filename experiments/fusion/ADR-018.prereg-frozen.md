# ADR-018 — Fusion: muons and gravity see one ore body (ADR-017 L5, first step)

**Status:** PRE-REGISTRATION. Sections 1–5 are frozen (SHA-256 in
`experiments/fusion/PREREG_SHA256.txt`, verbatim copy `ADR-018.prereg-frozen.md`) before any
`check_fusion` code is written or run. Results go in section 6 only.
**Date:** 2026-09-28
**Depends on:** ADR-017 (L1 plug-ins, reality-engine rules), `check_mine`, `check_gravity`.

## 1. Question

Gravity alone puts a body's undecided bits too shallow (`check_gravity`, a model depth bias).
Muons look up from tunnels and constrain depth through their angles. When both sensors' rows
are stacked on one `BitField`, does the posterior see the body better than either sensor
alone, and does gravity's depth bias shrink?

## 2. Scene (declared, not fitted)

- `check_mine`'s block, unchanged: rock 2650 kg/m³, |x|, |y| ≤ 36 m, z in [0, 48] m, 2 m cells,
  three upward-looking chambers at z = 6 m (x = −16, 0, 16), 3 m detector rooms surveyed.
- Ore: an 8 m cube (half-extent 4 m) at the origin in x and y, centre z = 28 m (20 m below the
  surface), density 1.25 × rock. Muon contrast a = −(μ_ore − μ_rock), and gravity contrast
  662.5 kg/m³, both from the same ratio.
- Muons: 2²³ candidates per chamber, lower than `check_mine`'s exposures, chosen so muons
  alone are expected to be weak.
- Gravity: 16 × 16 stations, 4 m apart, from −30 to 30 m, at z = 48.5 m; σ = 1 µGal, seed
  declared in code.
- Prior: p₀ = 10⁻³, λ = 2, κ = ln 999; the default posterior schedule; 16 branches.
- Each sensor's control is the same survey over the no-ore block, noise only.

## 3. Runs

Three problems on the same bits: **muons only**, **gravity only**, and **fused** (both
sensors' rows via `assemble`). Each gets a body run and a control run.

## 4. Predictions and exit rules

- **R1 (identity, must hold):** the fused data evidence equals the sum of the two sensors'
  data evidence, to 10⁻⁹ relative. The likelihoods are independent.
- **R2 (must hold):** no confident bit (p ≥ 0.9) in any of the three controls.
- **R3 (the question):** "claimed" bits are p ≥ 0.5. The fused depth error of the claimed-bit
  centroid must be smaller than gravity-only's. If either has no claimed bits, R3 is reported
  as not decidable, and that counts as a FAIL for the question, not for the check.
- **R4:** if the fused evidence pays, at least one confident bit, with its horizontal centroid
  within 4 m of the truth.

Exit 0 requires R1, R2 and R4. R3 is reported PASS / FAIL / not decidable in section 6 and is
not tuned.

## 5. What is not claimed

Density only, never chemistry (ADR-017 rule 1). No days of exposure (the flux is not
calibrated for this depth). Synthetic truth only.

## 6. Results

(to be filled after the run)
