# Machine export: three-state volume + traversable mask

What QuBLAR already emits for tooling, what was added for a loader, and
what KY-RO / KAUBY / TELIX / ndim actually consume. No retune of classify
(0.1 / 0.9), rooms, bar, anneal, or Blaze `rel_tol`.

## Files (under `build/`)

| File | Type | Meaning |
|---|---|---|
| `ising_out_pvoid.bin` | float32, C-order `(z,y,x)`, size `nx³` | Posterior P(void). **-1** = outside the unknown domain. Written by `check_ising`. **Do not duplicate.** |
| `ising_out.meta` | text | `nx`, `voxel`, `lo`, `branches`, `vars`, `lambda`, `kappa` |
| `ising_out_tristate.bin` | uint8, same shape | `0` Exists · `1` NotThere · `2` Undecided · `255` outside |
| `ising_out_traversable.bin` | uint8, same shape | `1` only if Exists (ahead policy); else `0` |
| `ising_out_machine.meta` | text | Loader keys for the two masks |

Derive the masks (does not re-anneal):

```bat
python tools\export_machine_map.py
python tools\export_machine_map.py --check-only
```

Sample the traversable mask along a voxel path (does not re-anneal):

```bat
python tools\walk_traversable.py
python tools\walk_traversable.py --face
python tools\walk_traversable.py --start 64 64 0 --step 0 0 1 --count 128
python tools\walk_traversable.py --start 64 64 56 --step 0 0 1 --count 16
python tools\walk_traversable.py --slice
```

Default path (declared, derived from the grid — not a screenshot pick):

| Quantity | Rule |
|---|---|
| start | centre of the in-domain mask (`p_void >= 0`): component-wise median of in-domain indices, snapped to the nearest in-domain voxel if needed |
| target | voxel of maximum `p_void` among in-domain (measured MAP of the posterior) |
| step | `sign(target − start)` per axis `(dz, dy, dx)`, one voxel at a time |
| line | axis-aligned → constant straight-step; otherwise 3-D Bresenham visiting each voxel on the segment |

Legacy outside-stop (former default), explicit CLI: `--face` or
`--start nx//2 nx//2 0 --step 0 0 1 --count nx`. On the existing export that
first voxel is outside (`p_void = −1`).

Indices are C-order `(z,y,x)`. Stops at the first voxel with `traversable != 1`.
Exit 0 when the Undecided→block invariant holds (a block is a result). Exit 1
on a leak, missing masks, or an unreported out-of-array step. Missing masks:
run `export_machine_map.py` first.

## Classification and policy (frozen)

From `ising_recon.hpp` `classify` and `check_ahead` / `RESULTS-capabilities.md`:

| State | Condition on `p_void` | Traversable? |
|---|---|---|
| Exists (rock) | `p ≤ 0.1` | yes |
| Undecided | `0.1 < p < 0.9` | **no** |
| NotThere (void) | `p ≥ 0.9` | no |
| outside | `p < 0` | no |

The evidence budget (data nats vs prior nats) is **not** inside these bins.
If data do not pay, `check_ising` / `check_ahead` decline the claim; the
masks still classify, but unpaid runs are not a green light to travel.

## Minimal loader (proof inside QuBLAR)

```python
import numpy as np
meta = dict(l.split(maxsplit=1) for l in open("build/ising_out.meta"))
n = int(meta["nx"])
p = np.fromfile("build/ising_out_pvoid.bin", np.float32).reshape(n, n, n)
trav = np.fromfile("build/ising_out_traversable.bin", np.uint8).reshape(n, n, n)
# invariant: every Undecided voxel blocks
assert ((p > 0.1) & (p < 0.9) & (trav != 0)).sum() == 0
```

`export_machine_map.py` runs that check (plus a synthetic `p=0.5` cell).

## What each machine actually consumes

Honesty: **none of KY-RO, KAUBY, or TELIX read a voxel posterior today.**
Do not fake a live bridge.

### KY-RO (`PR0JECTS/RESEARCH/KY-RO`)

- **Consumes:** `uint8` frames `H×W×C` via `PerceptionBackend.read()` →
  `detect_light` (HOG or blob) → centroid / presence features → gait steer
  ([`docs/ADR-006-perception.md`](../../KY-RO/docs/ADR-006-perception.md),
  [`src/kyro/perception/backend.py`](../../KY-RO/src/kyro/perception/backend.py)).
- **Also:** MuJoCo heightfields / XML meshes for sim terrain — geometry for
  physics, not an occupancy posterior.
- **Plug-in:** a QuBLAR three-state volume does **not** feed PerceptionBackend.
  A future planner could sample `ising_out_traversable.bin` along a path.
  QuBLAR now exposes that sampler as `tools/walk_traversable.py`; Kyro still
  does not call it.
- **Does not fit:** continuous camera state, no known scene truth in the loop,
  no forward model of muon transport.

### KAUBY (`PR0JECTS/RESEARCH/KAUBY`)

- **Consumes:** WLS mixer thrust commands / EDF saturation modes
  ([`PROGRAM.md`](../../KAUBY/PROGRAM.md), [`lab.html`](../../KAUBY/lab.html),
  [`PHYSICS.md`](../../KAUBY/PHYSICS.md)). CAD for the hex frame. No
  occupancy, voxel, point-cloud, or perception path.
- **Plug-in:** none. A traversable voxel grid is not a mixer input.
- **Does not fit:** no scene model, no prior over rock/void, no sensor rays.

### TELIX = TelomereSim (`PR0JECTS/RESEARCH/TelomereSim`)

- Name on disk: **TELIX** / `telix_*` (README title). Not a robot.
- **Consumes:** scenario parameters and per-cell state
  (telomere length, SSB load, TERT chain) → CSV / binary checkpoint
  ([`README.md`](../../TelomereSim/README.md),
  [`core/telix_model.hpp`](../../TelomereSim/core/telix_model.hpp)).
- **Plug-in:** none. QuBLAR volumes are spatial posteriors; TELIX is a
  population / kinetics simulator.
- **Does not fit:** no geometry, no occupancy, no forward imaging model.

### DIMMA (renderer, not a vehicle; was ndim-lab)

- **Consumes:** spacetime **point clouds** (`SpacetimeCloud`: rows of named
  axes, RGB) or LiDAR `.laz` / video
  ([`ndim/spacetime/cloud.py`](../../DIMMA/ndim/spacetime/cloud.py),
  ADR-0002/0003).
- **Already used:** QuBLAR `tools/render_branches.py` converts the exported
  tri-state voxels into a point cloud of centres and renders with ndim.
  That is a **view of the measured posterior**, not a world-model video.
- **Does not fit as a planner:** ndim does not navigate; it projects / sections.

## Path (measured)

Command (default path on the existing `build/` export, nx=128):

```bat
python tools\walk_traversable.py --slice
```

| Quantity | Value |
|---|---|
| start (in-domain median centre) | `(19, 63, 63)` |
| target MAP (max `p_void`) | `(43, 59, 63)` |
| step rule | `sign(Δ)=(1,−1,0)`; Bresenham (not axis-aligned) |
| leaks (Undecided with traversable=1) | **0** |
| first block `(iz,iy,ix)` | `(43, 59, 63)` |
| class | **NotThere** |
| p_void | **1.0** |
| inside pyramid (`p_void >= 0`) | **yes** |
| exit | 0 (invariant holds; block is a result) |

Legacy face path (reproducible outside-stop):

```bat
python tools\walk_traversable.py --face
```

| Quantity | Value |
|---|---|
| start / step / count | `(64,64,0)` / `(0,0,1)` / `128` |
| first block | `(64, 64, 0)` · **outside** · `p_void=−1.0` |

On this volume the unknown domain on the centre line `z=y=64` is only
`ix∈[56,71]`; the face start is outside padding. The default walk starts at
the median centre of the in-domain mask and stops on the measured MAP void.
Slice of the measured `p_void` field (y-slice at the walk start `iy=63`):
`docs/figures/walk_traversable_slice.png`. Caption states that exposure /
evidence-budget pay is **not** stored in the bins.

## What still does not fit (all three vehicles)

1. **No consumer of the voxel posterior** in KY-RO / KAUBY / TELIX source trees
   (QuBLAR can walk its own mask; those repos do not load it).
2. **No continuous robot state** in QuBLAR’s answer — only bits / tri-state.
3. **Known-truth scoring** is QuBLAR’s closed-loop discipline; the robots do
   not supply a ground-truth void for that contract.
4. Inventing a frame synthesizer or mixer shim would fake a bridge. Declined.
