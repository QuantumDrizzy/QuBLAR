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

## 5. Results

(to be filled after the re-runs)
