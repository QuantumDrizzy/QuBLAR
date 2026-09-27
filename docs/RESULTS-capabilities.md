# Capability checks: mine ore and look-ahead cavity

Measured on the RTX 5060 Ti host with `build\check_mine.exe` and
`build\check_ahead.exe`. Thresholds, schedule, classify (0.1 / 0.9), 3 m detector
rooms, and the 6 m localisation bar (only when data pay) are the frozen void-path
defaults from `check_ising` / `ising_recon.hpp`. They were not retuned.

**Days are not claimed.** Exposure is reported as candidate counts per chamber.
Lesparre et al. (GJI 183, 1348, 2010) §4 give T ≈ 15 cm² sr → ~1000 muons/day at
~100 m of standard rock and ~1 muon/day at ~1000 m, via N = I · T · days. The
code’s flux model was not inverted for I at these synthetic depths, so no
day-count is published here.

## Declared inputs (not fitted)

| Quantity | Value | Cite / note |
|---|---|---|
| Host rock density | 2.65 g/cm³ | Lesparre / PDG “standard rock” |
| Ore density ratio | ρ_ore / ρ_rock = 1.25 | Midpoint of owner’s 20–30% brief; **not** a published ore grade |
| μ_rock | 0.046 /m | ADR-006 §1 (engine constant) |
| μ_ore | 0.0575 /m | ADR-006 §1: μ ∝ ρ → μ_rock × 1.25 |
| a_per_metre (ore QUBO) | −0.0115 /m | −(μ_ore − μ_rock); same deficit form as voids |
| Prior | p₀ = 10⁻³, λ = 2, κ = ln(999) | Same as void path in `check_ising` |
| Anneal schedule | default `Schedule` (200 + 50, T: 50 → 1) | Not lengthened after scores |
| Ahead distance | 5 m | Declared “few metres” |

## Row 1 — `check_mine` (denser ore under rock)

Synthetic host rock block, three upward-looking chambers, one denser ore body.
Control scene (no ore) must not invent a confident ore body. Same schedule for
both body sizes. Exit 0 on a correct decline or a correct paid localisation.

| ore half (cube) | candidates / chamber | data nats | prior nats | budget | confident ore (correct) | dist to centre | control ore | exit |
|---|---|---|---|---|---|---|---|---|
| 2.0 m (~4 m) | 2²⁵ | +74.3 | +103.3 | **DECLINE** | 0 (0) | — | 0 | 0 |
| 2.0 m (~4 m) | 2²⁶ | +185.1 | +103.3 | pay | 4 (4) | 1.00 m | 0 | 0 |
| 2.0 m (~4 m) | 2²⁷ | +414.6 | +103.3 | pay | 8 (8) | 0.00 m | 0 | 0 |
| 4.0 m (~8 m) | 2²⁵ | +1603.6 | +634.0 | pay | 47 (47) | 0.96 m | 0 | 0 |
| 4.0 m (~8 m) | 2²⁶ | +3419.1 | +634.0 | pay | 57 (57) | 0.42 m | 0 | 0 |
| 4.0 m (~8 m) | 2²⁷ | +7010.7 | +634.0 | pay | 63 (63) | 0.07 m | 0 | 0 |

The ~4 m cube (half-extent 2.0 m) at 2²⁵ is a **correct decline** (gain 74.3 ≤ cost 103.3): no confident
true-ore label, control clean, exit 0. That is not a failure. The same body pays
from 2²⁶. The ~8 m cube pays at every measured exposure.

## Row 2 — `check_ahead` (cavity 5 m ahead)

Policy on the existing tri-state map (`classify` unchanged):

- Exists (rock, p ≤ 0.1) → traversable
- Undecided or NotThere → not traversable

When data do not pay: no confident void on the truth (same rule as 2²⁵ void);
localisation is not claimed. When data pay: every truth cavity voxel must block,
and the NotThere centroid must lie within 6 m of the cavity centre. Empty control
must stay clear and invent no confident void.

| candidates | data nats | prior nats | budget | NotThere on truth | clear on truth | dist m | control voids | exit |
|---|---|---|---|---|---|---|---|---|
| 2²⁰ | +480.7 | +634.0 | **DECLINE** | 0 | 64 (Exists) | — | 0 | 0 |
| 2²⁴ | +6114.3 | +634.0 | pay | 11 | 29 | 1.64 | 0 | **1** |
| 2²⁵ | +12332.8 | +634.0 | pay | 14 | 0 | 0.95 | 0 | 0 |
| 2²⁶ (default) | +24546.9 | +634.0 | pay | 4 | 0 | 0.79 | 0 | 0 |

At 2²⁰ the posterior correctly **declines** the cavity (exit 0). Prior-dominated
Exists on the truth is expected when unpaid; the policy map still treats Undecided
as blocking, but unpaid runs are not required to produce Undecided.

At 2²⁴ the evidence budget says the data pay and a NotThere set localises within
6 m, but 29 of 64 truth cavity voxels remain Exists (would green-light travel into
the hole). Exit 1. Filed, not retuned.

From 2²⁵ the paid policy holds: all truth cavity voxels block, localisation < 6 m,
control clean.

## Local field

Host check `build\check_local_field.exe`: on a 6-variable synthetic QUBO,
`local_field_dE` matches `binary_energy(after) − binary_energy(before)` for
every variable under all-rock, mixed, and all-void (18 flips, both directions,
including at least one uphill `dE > 0`). Worst abs err **8.88e-16** (ROI-style
bar `1e-6 · max(1, |E|)`; measured scale ≈ 7.50). Exit 0.

The accept/reject stays in C++ because it branches on the data and on a random
draw; LYTH does not compile that.

## Build

```bat
cmd /c "call \"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat\" >nul && nvcc -O3 -arch=sm_120 -std=c++17 -I src -o build\check_mine.exe src\check_mine.cu && nvcc -O3 -arch=sm_120 -std=c++17 -I src -o build\check_ahead.exe src\check_ahead.cu && nvcc -O3 -arch=sm_120 -std=c++17 -I src -o build\check_local_field.exe src\check_local_field.cu"
```

Optional exposure: `check_mine.exe <log2 N>`, `check_ahead.exe <log2 N>`.

## GARY meter

QuBLAR emits the bits. Blaze compresses them only on a compressed verdict. GARY
asks whether the dependence survives a shuffle — it is not the motor.

Host check `GARY\build\Release\gary_qublar_bits.exe` on the 6-bit fixture
(mass 0.5 on `000000` and 0.5 on `001100`): structured
`I(bit2;bit3) = 1` bit; shuffle null (product of marginals) `I = 0` bits
(within phase-0 tolerance `1e-9`). Exit 0.
