# ADR-005 — External validation: real confocal data, and two wave-based reconstructions

**Status:** accepted
**Date:** 2026-09-17
**Implements:** the phase that ADR-004's action items left open: the reconstruction
baselines exist and are truth-scored, but nothing in this repository has ever touched a
measurement made by someone else.

## Context

Every number in this repository so far is internally validated: the simulator against
closed forms and statistical laws, the reconstructions against the truth the simulator
emits. That is the right design for phases 1–3 and it is also a closed loop. A reviewer's
first question — "does any of this agree with reality?" — has no answer inside the repo.

In 2018, O'Toole, Lindell and Wetzstein published *Confocal Non-Line-of-Sight Imaging
Based on the Light-Cone Transform* (Nature 555) and released both their capture data and
their MATLAB reconstruction code. Their rig is confocal — the same transport QuBLAR's
`transient_trace` simulates — which makes their data the cheapest possible external
contact: no hardware, no capture campaign, and a published reference pipeline to port.

### What was measured before deciding

A staged probe, in the spirit of ADR-002, because each fact would have reshaped the
design:

| | |
|---|---|
| Availability | All four release links (supplemental PDF, LCT code+data, iterative LCT, simulated bunny) still resolve; downloaded 2026-09-17. |
| Contents | `cnlos_reconstruction.m` plus **11 captured scenes**: resolution/dot charts at 40 and 65 cm, mannequin, exit sign, "S U" letters, outdoor S, diffuse S, simulated bunny. |
| Data format | MATLAB v5 `.mat`, readable by `scipy.io`. Each file: `rect_data` — uint16 counts, shape `[N, N, M]` with `N = 64` spatial (256 for the simulated bunny), `M = 2048` time bins (1024, bunny) — and `width`, the half-extent of the scanned wall patch (0.35 m; 0.5 m bunny). |
| Time base | Native bin 4 ps (8 ps, bunny). Data is **pre-rectified**: the direct wall return starts at the first time bin, per-pixel shifted, and cropped. |
| Their pipeline | Downsample 2× twice (→16 ps), zero the first bins (direct gate), build the light-cone PSF, Wiener-filter in 3D FFT domain, resample t→√t (sparse operator, built at native resolution then averaged down), pad to `[2M, 2N, 2N]`, FFT³·filter·IFFT³, resample depth axis, clamp. |
| Quirks that must survive a port | 1-based `linspace(-1,1,2U)` endpoints (the "centre" index is not exactly zero), `circshift` of the PSF by `U` along both spatial axes, `c = 3e8` m/s (we use `kSpeedOfLight`; 0.07% difference, mirrored in the reference port so the comparison is unaffected). |

## Decision

### 1. The external data enters as a converted artefact, never as MATLAB in the hot path

`tools/mat_to_raw.py` (Python is sanctioned by ADR-001 for dataset packaging, never the
hot path) converts each `.mat` to `data/ext/<scene>.bin` (float32, `[N, N, M]` natural
order) plus `data/ext/<scene>.meta` — a flat `key = value` text file, **not** JSON,
because parsing JSON in C++ would add a dependency this repository does not take. The
`.meta` records grid, native bin, width, their per-scene gate and display offsets, the
diffuse flag, and the source SHA-256. `data/` is git-ignored; the converter is committed.

### 2. An FFT is written, not linked

LCT and phasor field both need FFTs; the repository links nothing outside CUDA/OptiX and
keeps it that way. `src/fft.hpp`: iterative radix-2 complex FFT (host-only, float),
sizes asserted to powers of two — every size in this pipeline is (`64/128/256/512/1024`).
Validated against a direct DFT, against Parseval, and against a closed form (a cosine
window must land its two peaks at ±k with unit ratio). The 3D transforms are separable
passes of the 1D one.

### 3. The loader speaks `nlos.hpp`'s layout, so backprojection runs on real data unmodified

`src/external.hpp` reads bin+meta and emits `n_relays × bins` transients plus the relay
grid, i.e. exactly what `laplacian_filter`/`backproject` already consume. One addition
to `nlos.hpp`: `backproject` takes an optional per-relay `L1` override, defaulting to
"compute from sensor position". The released data is rectified — time already excludes
the sensor↔wall leg — so for it `L1 = 0` and the spheres are centred on the relay points
themselves. This is a consequence of their calibration, not a special case invented for
convenience, and the default path is unchanged.

### 4. Two new methods, both host-only, both inversions of the model already validated

- `src/lct.hpp` — a port of the released confocal LCT pipeline (Wiener deconvolution of
  the light-cone PSF, √t resampling, 3D FFT). It is **the authors' algorithm**, not ours;
  agreement with their published figures validates the port, and QuBLAR's job is to run
  it next to truth-scored baselines.
- `src/phasor.hpp` — phasor-field reconstruction (Liu et al., Nature 2019): the gated
  transients are the time-varying field on the wall; per temporal frequency ω the field
  propagates into the volume by angular spectrum, and the per-frequency fields sum
  coherently. The virtual wavelength falls out of the band chosen around the system
  response, and the band and its sensitivity sweep are parameters named in code, because
  the released data does not carry the rig's true impulse response.

Both are host-only on purpose — ADR-003's rule: the yardstick does not live on the GPU
with the thing it measures.

### 5. Two independent port checks, stated before any of them runs

- A **numpy mirror** of the released MATLAB pipeline (`tools/lct_reference.py`) runs on
  the same scenes; the C++ LCT must agree with it numerically. Two ports written from
  the same source by the same author catch transcription slips, not conceptual errors —
  that is all they are claimed to catch.
- The **synthetic replica**: QuBLAR simulates a confocal scan of a wall patch with the
  released geometry (0.7 m patch, 64×64 relays, 4 ps bins rectified the same way), with
  hidden patches whose positions are the truth. The same three reconstructions —
  backprojection, LCT, phasor — run on synthetic and real data through one code path.

### 6. Scoring: truth where truth exists, honesty where it does not

- **Synthetic (truth exists):** peak localisation error per method, and volume
  precision/recall against occupied truth voxels — reported separately, never combined
  (the ADR-003 rule; a method can buy precision with an empty volume and recall with a
  full one). The empty-room control stays: a reconstruction that answers confidently in
  an empty room is reconstructing its own gate.
- **Real (no transport truth exists):** cross-method consistency (distance between
  per-method peaks and between energy centroids) and visual comparison against the
  paper's published figures, labelled as visual. Nothing more is claimed.

### 7. What the numbers may and may not say

- Agreement of the C++ LCT with the numpy mirror and with published figures validates
  **the port and the pipeline end to end** (loading, units, gating, FFT, resampling).
- Truth-scored results on the replica validate **the reconstruction methods against the
  transport model** — which is the thing only QuBLAR can do.
- The gap between the two — where a method wins on synthetic and loses on real data —
  localises unmodelled physics (wall BRDF, jitter tail, ambient spectrum) but does not
  attribute it to a cause. Any sentence of the form "the difference is because of X"
  requires a follow-up experiment that toggles X in the simulator. This is written here
  so RESULTS-phase4 cannot accidentally say it.

### 8. A missing dataset is not a green run

`check_external` always runs its FFT self-checks; the data section reports
`SKIP (no data)` and counts as a failure. A machine without `data/ext/` cannot produce
an "all green" it did not earn.

## Consequences

- The reconstruction yardstick gains two methods that share no code path with the
  forward model, which is the first real test of the ADR-004 claim that the baselines
  check assumptions rather than share them.
- `data/` (1.4 GB of downloads) stays out of git; a fresh clone runs everything except
  the external section until the converter is run.
- The bunny scene is simulated by the authors themselves; it is loaded as a third,
  clearly-labelled category — neither captured nor QuBLAR-generated — and is excluded
  from "real data" claims.
- Host-side 3D FFT on `[2M, 2N, 2N]` (real scenes: 1024·128·128 complex floats) is a few
  hundred MB and seconds; acceptable, and noted as the ceiling if scenes grow.

## Action items

1. [x] `tools/mat_to_raw.py` + `tools/lct_reference.py`; convert the captured scenes
2. [x] `src/fft.hpp` with DFT / Parseval / closed-form checks, run in every check pass
3. [x] `src/external.hpp` loader; `backproject` gains the rectified-data `L1` override
4. [x] `src/lct.hpp` — port of the released pipeline, checked against the numpy mirror
5. [x] `src/phasor.hpp` — angular-spectrum phasor field, band sweep as a named parameter
6. [x] `src/check_external.cu` — FFT checks, real-data section (SKIP=fail without data),
       synthetic replica with truth, empty-room control, three methods, both scorings
7. [x] build.bat / check.bat / sanitize.bat / .gitignore / README updates
8. [x] `docs/RESULTS-phase4.md` written after measurement, claims scoped per §7

Measured outcome: `docs/RESULTS-phase4.md`.

> **Amended 2026-09-18, after measurement.** Two decisions made during
> implementation are recorded here rather than silently: (1) the C++ LCT is
> scored against a golden volume dumped by `tools/lct_reference.py
> --dump-full` (8.4 MB, git-ignored, regenerable) -- a port this size needs a
> numeric referee, and the mirror is it; (2) the phasor field is implemented
> and runs but does not yet meet its localisation bar on the replica, so its
> replica checks carry a strict XFAIL status in `check_external` -- visible,
> non-blocking, and failing loudly if they unexpectedly pass. Peak reporting
> on real data uses the authors' own display window, computed from their
> released per-scene parameters. The measured numbers, including the phasor's
> open status, are in RESULTS-phase4.md.
