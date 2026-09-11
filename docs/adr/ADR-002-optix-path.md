# ADR-002 — The OptiX path, and what its speedup is allowed to claim

**Status:** accepted
**Date:** 2026-09-11
**Supersedes:** nothing. Implements ADR-001 action items 8 and 9.

## Context

ADR-001 built a hand-written CUDA BVH first, on the explicit reasoning that claiming "RT
cores are N× faster" requires something to be N× faster *than*. That baseline now exists
and is green: 9 correctness invariants, including watertightness.

This ADR decides how the RT-core path is built alongside it, and — more importantly —
what the resulting number may and may not be said to mean.

### What was measured before deciding

Four facts were established by a staged probe rather than assumed, because each one would
have silently reshaped the design:

| | |
|---|---|
| OptiX SDK | 9.1.0, headers only; the implementation is loaded from the driver at runtime |
| RTCORE_VERSION | 40 — the RT core generation on this Blackwell part |
| **OptiX under WSL** | **does not work.** `optixInit` returns `LIBRARY_NOT_FOUND`; with `LD_LIBRARY_PATH` pointed at `/usr/lib/wsl/lib` it returns `ENTRY_SYMBOL_NOT_FOUND`. `libnvoptix.so.1` there is **14 KB** — a stub, not an implementation. This is the same class of gap that makes `ncu` fail under WSL on this machine. |
| module format | `nvcc --ptx` is unusable: CUDA 13 validates the result with `ptxas`, which rejects the OptiX intrinsics (`_optix_trace_typed_32` and friends) as unknown symbols. `--optix-ir` bypasses ptxas and works. |

The Windows host link also needs `advapi32.lib`: OptiX's loader reads the registry to
locate the driver DLL.

## Decision

### 1. The comparison is rays in, hits out — and nothing more

The OptiX path implements the *same interface* as `traverse_bvh`: a batch of rays, a batch
of nearest hits, identical struct layouts, identical scene.

It deliberately does **not** reimplement the sensor model, for a concrete reason: OptiX
programs cannot use `__shared__` memory, and the whole point of the LiDAR kernel's design
is that a beam's waveform is accumulated in shared memory. Porting it would force that
accumulation into global memory, and the resulting comparison would be dominated by the
memory system rather than by the ray-geometry engine. That is not the question being
asked.

**Consequence for the claim, stated before any number exists:** the measurement is a
statement about ray-geometry queries only. It is an *upper bound* on what the full sensor
would gain, not a prediction of it, because the sensor does per-sample radiometry, pulse
splatting and truth accumulation that no tracer speedup touches. Any sentence of the form
"QuBLAR is N× faster with RT cores" is therefore wrong; the true sentence names the
traversal.

### 2. One toolchain for both sides

Everything moves to a Windows build under `vcvars64`. The existing WSL flow works fine for
the CUDA-only binaries, so this is a change to something that is not broken — and it is
still correct, because OptiX cannot run under WSL and the baseline and the accelerated
path must not be built by different compilers.

WSL has nvcc 12.8; the Windows host has 13.0. A speedup measured across that boundary
would carry a confound that no amount of repetition removes, and the first reasonable
question anyone would ask is whether the baseline was simply compiled worse. Both sides,
one compiler, one set of flags.

### 3. Agreement before timing, and the tolerance is not zero

The two tracers must agree before either is timed, on the same tests the baseline already
passes. But they will **not** agree bit for bit, and pretending otherwise would mean
tuning a tolerance until the test goes green:

- OptiX's triangle intersection is performed by fixed-function hardware using a watertight
  formulation that is not Möller–Trumbore. Distances agree to floating-point tolerance,
  not exactly.
- Silhouette-edge hits may differ in which triangle is reported when a ray strikes a
  shared edge. Both answers are correct; the reported distance must still match.

So the agreement criteria are: hit/miss must match exactly, reported distance must match
within a declared tolerance, and the surface actually described must match. Where they
disagree, the disagreement is reported rather than absorbed.

Running the watertightness test through OptiX is a cross-check worth having in its own
right: the hardware intersector is watertight by construction, so a leak there would mean
the *scene* is open rather than the intersector, and that would invalidate the baseline's
result too.

### 4. The speedup is a curve, not a number

A single ratio is measured on a single scene and hides the thing that actually matters:
RT cores win more as the hierarchy deepens. A benchmark on the 432-triangle box would
mostly measure launch overhead.

So the harness sweeps triangle count, and reports the ratio at each size. It also reports
**coherent and incoherent ray distributions separately**, because the gap between them is
large for both implementations and a benchmark that quietly uses only coherent rays is
the standard way this number gets inflated.

Build times are reported separately and never folded in: OptiX's acceleration-structure
build is hardware-assisted, the baseline's is a host-side recursion, and adding them to
the traversal time would flatter whichever side one wished to favour.

### 5. Timing methodology

CUDA events around the launch only, excluding upload and allocation. Warm-up launches
discarded. Repeats reported as a **median**, not a mean — a single scheduler preemption
moves a mean and does not move a median. Both implementations get identical ray buffers in
identical order.

## Consequences

- Phase 1's binaries must be rebuilt and re-verified under the Windows toolchain before
  any of this is trusted. A green run under WSL says nothing about the Windows build.
- The repository gains a `build.bat`. Batch files must be written with CRLF; an LF-only
  `.bat` fails with "not recognized as an internal or external command", which reads like
  a missing file and is not.
- `libnvoptix.so.1` under WSL being a stub is worth remembering beyond this project: the
  WSL GPU layer on this machine ships the CUDA core and stubs some of the rest.
- If OptiX ever becomes awkward, Vulkan ray queries reach the same hardware. Not needed
  now, and noted so the dependency is a choice rather than an assumption.

## Action items

1. [x] Port the Phase 1 build to Windows; re-verify all 20 checks there
2. [x] OptiX device programs: raygen, miss, closesthit writing the same `Hit`
3. [x] Host pipeline: module, program groups, SBT, acceleration structure
4. [x] Agreement harness against the baseline, including watertightness
5. [x] Timing sweep over triangle count, coherent and incoherent rays
6. [x] Record the result with the claim scoped to traversal

Measured outcome: `docs/RESULTS-phase2.md`.
