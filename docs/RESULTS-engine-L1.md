# L1, the model layer: first results (ADR-017)

Measured on the RTX 5060 Ti host, 2026-09-28. Binaries: `build\check_engine.exe`,
`build\check_gravity.exe`.

## What L1 changed

The engine used to be soldered to muons: `build_binary_problem` traced rays inside itself. It
is now three plug-ins (`src/engine.hpp`) assembled into the same `BinaryProblem` that the
annealer, the freezer and the ROI already use:

- **`BitField`** — the bits, the site each one lives at, and their adjacency;
- **`OperatorRows`** — one sensor's rows: `d`, `w`, and each bit's share `a`;
- **`IsingPrior`** — κ and λ.

Muons are the first sensor (`src/op_muon.hpp`, the old code moved unchanged). Gravity is the
second (`src/op_gravity.hpp`). `build_binary_problem` is now three lines.

## check_engine: all pass

| Check | Result |
|---|---|
| A. regression against the frozen pre-L1 builder, void path | identical, array for array: 306,360 bits, 55,257 rows, 2,972,871 entries |
| A. regression, denser-body path | identical |
| B. adjoint, muon sensor | relative mismatch 1.42 × 10⁻¹⁷ |
| B. adjoint, gravity sensor | relative mismatch 1.14 × 10⁻¹⁶ |
| C. gravity: a sphere of cells vs a point mass at its centre (shell theorem) | worst 0.008 % over 3 stations, 912 cells |
| D. muons + gravity on one field | rows stack (55,257 + 64), and E(fused) = E(muons) + E(gravity) − prior exactly |

About C: the reference uses the cells' own total mass, so the test isolates how the mass is
distributed, not the volume discretisation. Cubic symmetry cancels the low-order corrections,
which is why the error is small.

Because A holds array for array, every existing muon check (`check_ising`, `check_mine`,
`check_ahead`) sees the same problem it saw before L1. They were not re-run on the GPU for this
note.

## check_gravity: a second physics through the same engine

The scene uses no muons: 16,384 bits, 256 gravimeter stations and a denser body of 64 bits (an
8 m cube, centre 12 m down, contrast 662.5 kg/m³). All inputs were declared in the file header
before the first run.

| σ (µGal) | data (nats) | prior (nats) | budget | confident bits | undecided | control | verdict |
|---:|---:|---:|---|---:|---:|---:|---|
| 5 | 50.7 | 634.0 | decline | 0 | 0 | 0 | PASS |
| 2 | 382.7 | 634.0 | decline | 0 | 31 | 0 | PASS |
| 1 | 1588.6 | 634.0 | pay | 0 | 85 | 0 | **XFAIL: depth bias, model** |

The bits at σ = 1 µGal, a section through the body. The legend is `1` body, `0` rock, `?` undecided:

```
  depth   truth                              QuBLAR
    3 m  00000000000000000000000000000000   0000000000000000000??00000000000
    5 m  00000000000000000000000000000000   00000000000000?????0000000000000
    7 m  00000000000000000000000000000000   0000000000000??????0000000000000
    9 m  00000000000000111100000000000000   00000000000000??????000000000000
   11 m  00000000000000111100000000000000   00000000000000000000000000000000
   13 m  00000000000000111100000000000000   00000000000000000000000000000000
   15 m  00000000000000111100000000000000   00000000000000000000000000000000
```

**What it says.** Horizontally the engine is on the body. It never claims a body it cannot
support: no confident bit anywhere, a clean control, and a correct decline at 5 and 2 µGal. But
its undecided bits sit **too shallow**, at 3–9 m, where the truth is at 9–15 m.

**Why: the model, not the sampler.** At σ = 1 µGal, E(truth) = 758.0 and the best branch is
641.8: the posterior genuinely prefers shallower, smaller configurations. A cell's signal
falls with distance and the prior charges every bit the same, so a few shallow bits explain
the data more cheaply than the real body. This is gravity's known depth bias (Li & Oldenburg,
1998, *3-D inversion of gravity data*, Geophysics 63). The row is an XFAIL with that exact
signature: data pay, no confident bit, clean control, E(best) < E(truth). Any other failure,
or a pass, is reported as what it is.

## Next, pre-registered before anything is tried

1. **A depth-weighted prior** as an L1 prior plug-in: a per-bit linear field from Li &
   Oldenburg's depth weighting, with its exponent taken from the paper and not fitted here. The
   XFAIL row is re-run unchanged.
2. **Fusion, muons + gravity** (L5, already possible after L1): muons constrain depth through
   their angles. The prediction is that the depth bias shrinks when both sensors see the body.
   Written down now, and not yet run.
