# ADR-019 — The universal rule: no confident bit off the truth

**Status:** PRE-REGISTRATION. Sections 1–4 are frozen (SHA-256 in
`experiments/u1/PREREG_SHA256.txt`, verbatim copy `ADR-019.prereg-frozen.md`) before any check
is edited or re-run. Results go in section 5 only.
**Date:** 2026-09-28
**Depends on:** ADR-017 (reality-engine rules), ADR-018 §6 (the finding that prompted this).

## 1. Why

In ADR-018, muons alone at 2²³ declined the ore and still marked **3 confident bits, none on
the ore**. That is a stone called ore. No check caught it. Every check counted its confident
bits and its correct bits, but none required the two to be equal. The rules looked only at
the centroid, or only at the body. A reality engine's first discipline is not to see what is
not there, so this becomes a rule of every check with synthetic truth.

## 2. The rules

- **U1 (hard, exit-changing).** In every run with synthetic truth, body runs and controls
  alike, no bit may be confident (p ≥ 0.9 on the anomaly side) where the truth has no
  anomaly. Confident anomaly bits must equal confident correct bits.
- **U2 (reported, not exit-changing).** Confident misses are counted and printed: bits on
  the true anomaly with p ≤ 0.1. Under a sparsity prior, a body the data do not pay for is
  expected to be called host rock. That is the decline working, and it is reported, not
  hidden.

## 3. Scope, and what does not change

- **Scope:** `check_ising` (section B, the body run), `check_mine` (each body size and
  exposure), `check_ahead`, `check_gravity` and `check_fusion` (each sensor and the fused
  run).
- **Unchanged:** priors, thresholds (0.1 / 0.9), schedules, exposures, scenes and seeds. No
  number in any check moves to make U1 pass.
- A U1 violation is a FAIL of that check. It is reported with its count and never waived.
  A fix needs its own pre-registered ADR.

## 4. Known before the re-run

`check_fusion`'s muons-only run has 3 confident bits off the ore (ADR-018 §6), so under U1
`check_fusion` is expected to FAIL on that row. No other outcome is predicted.

## 5. Results (2026-09-28; sections 1–4 unchanged, no number in any check moved)

"Off truth" means confident anomaly bits where the truth has none (U1). "Misses" means
anomaly bits called host with confidence (U2, reported only).

| check | run | budget | confident (correct) | off truth | U1 | misses (U2) | exit |
|---|---|---|---:|---:|---|---:|---|
| `check_ahead` | cavity 5 m ahead, 2²⁶ | pay | 4 (4) | 0 | PASS | 0 of 64 | 0 |
| `check_gravity` | σ = 5 µGal | decline | 0 (0) | 0 | PASS | 64 of 64 | 0 |
| `check_gravity` | σ = 2 µGal | decline | 0 (0) | 0 | PASS | 64 of 64 | 0 |
| `check_gravity` | σ = 1 µGal | pay | 0 (0) | 0 | PASS | 48 of 64 | 0 (XFAIL depth bias, unchanged) |
| `check_fusion` | muons only, 2²³ | decline | 3 (0) | **3** | **FAIL** | 64 of 64 | **1** |
| `check_fusion` | gravity only, σ = 1 | decline | 0 (0) | 0 | PASS | 64 of 64 | |
| `check_fusion` | fused | pay | 20 (20) | 0 | PASS | 17 of 64 | |
| `check_ising` | pyramid void (section B) | pay | 11 (11) | 0 | PASS | 0 of 32 | 0 |
| `check_mine` | 4 m cube, 2²⁵ | decline | 0 (0) | 0 | PASS | 0 of 8 | 0 |
| `check_mine` | 4 m cube, 2²⁷ | pay | 8 (8) | 0 | PASS | 0 of 8 | 0 |
| `check_mine` | 8 m cube, 2²⁵ | pay | 47 (47) | 0 | PASS | **16 of 64** | 0 |
| `check_mine` | 8 m cube, 2²⁷ | pay | 63 (63) | 0 | PASS | 0 of 64 | 0 |

Controls are unchanged: 0 confident anomaly bits in every control of every check.

**Verdict.** U1 holds in every run but one: `check_fusion`'s muons-only row, as predicted in
§4. `check_fusion` now exits 1, and it stays that way until a pre-registered fix passes. Every
earlier claim of `check_ising`, `check_mine`, `check_ahead` and `check_gravity` survives the
rule. Their confident bits were all on the truth.

**What U2 shows, labelled as observation, not rule.** On a decline, a sparse prior calls the
body host rock with confidence, as §2 expects: gravity σ = 5 and 2, and every declining sensor
in `check_fusion`. Two paid runs also carry confident misses, and that is not a decline at
work:
- `check_mine`, 8 m cube at 2²⁵: 16 of 64 ore bits at p ≤ 0.1, while 47 are confidently ore.
- `check_fusion`, fused: 17 of 64 ore bits at p ≤ 0.1, while 20 are confidently ore.

In both, the data pay for the body, yet part of it is called rock with confidence. Where those
bits sit in the body is not yet measured (edges or depth). Calibration (ADR-017 rule 3) is
where this is to be tested: a p ≤ 0.1 bit should be ore at most about 10 % of the time, here on
paid bodies. That is not measured yet and is not claimed.

**Open, each needing its own pre-registration:**
1. The muons-only U1 failure: 3 bits about 11 m below the ore, on the line of near-vertical
   rays. Candidates are the depth-weighted prior or a ray-coverage floor. Nothing is tuned here.
2. Confident misses on paid bodies: calibration per decile on body bits.
