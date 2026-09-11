# ADR-001: A ground-truth LiDAR simulator on ray-tracing hardware

**Status:** Proposed
**Date:** 2026-09-11
**Deciders:** Antonio (QuantumDrizzy)
**Target machine:** RTX 5060 Ti, sm_120 (Blackwell), 36 SMs, 32 MiB L2, CUDA 13.3

## Context

A conventional LiDAR reduces each returned pulse to a single number: range. Everything
else in that return is discarded — and everything else is where the information is.

What a full return actually carries:

| in the waveform | what it encodes |
|---|---|
| multiple peaks | several surfaces inside one beam footprint: foliage, glass, fog, edges |
| peak width | surface roughness and incidence angle, since a tilted plane spreads the return in time |
| peak amplitude | reflectance, attenuated by 1/d² |
| arrival statistics | at single-photon level, the full transient — which is what non-line-of-sight and through-scattering imaging reconstruct from |

The obstacle to working on any of that is not the algorithms. It is that **you cannot
validate a reconstruction without knowing the answer**, and on real hardware you never do:
a real scan gives you a waveform and an opinion about what produced it.

So the first artifact is not a reconstruction algorithm. It is a generator that produces
waveforms *and* the exact geometry that produced them.

### The hardware argument

Simulating a LiDAR is ray tracing. Each pulse is a cone of rays; each ray needs its
nearest intersection with the scene, its distance, its surface normal and its material.
That is precisely and exactly what an RT core does, in dedicated silicon, at a rate no
hand-written CUDA traversal will approach.

That silicon is idle on this machine. `nvidia-smi` reports Blackwell with RT cores
present; the OptiX SDK is not installed and nothing in the existing lab has ever issued a
ray. The whole field treats ray tracing as a rendering technology. It is a
geometry-query engine, and sensing is a geometry-query problem.

### Forces

- Ground truth is the entire point. Any physics the simulator gets wrong becomes a
  reconstruction result that is wrong in a way nothing can detect.
- The OptiX SDK is behind an NVIDIA developer login, which is a manual step.
- This inherits `rse-hpc-lab`'s discipline: a number that no probe produced does not get
  rendered, and a speedup needs something real to be measured against.

## Decision

Build **QuBLAR**: a photon-transport LiDAR simulator that emits, for every beam,

1. the **measurement** — a time-resolved return, in one of two detector models, and
2. the **truth** — the exact ranges, normals and material ids of every surface the beam
   actually struck, with the per-surface contribution that produced each feature.

Reconstruction algorithms are then scored against (2) rather than against each other.

### The physics, and what is deliberately not modelled

A beam is not a ray. It has angular divergence, so its footprint at range covers an area
that may contain several surfaces at different distances. The return is

    r(t)  =  Σ over sampled rays  [ p(t − 2d/c) · ρ · cos θ / d² ]  +  background

where `p` is the transmitted pulse shape (Gaussian, nanosecond-scale), `ρ` the surface
reflectance at the laser wavelength, `θ` the incidence angle, `d` the range. Monte Carlo
sampling over the footprint is what the RT cores do; the rest is what we do with the hits.

Two detector models, because they are different instruments and behave differently:

- **Linear-mode (APD).** The waveform above, plus additive noise. What commercial
  full-waveform systems record.
- **Photon counting (SPAD / TCSPC).** Photon arrivals are a Poisson process whose rate is
  proportional to `r(t)`. Arrivals accumulate into a histogram over many pulses. This model
  must include **dead time**, because a SPAD that has just fired is blind for tens of
  nanoseconds — which biases early photons and distorts the histogram (*pile-up*). A
  simulator without pile-up produces data no real detector could produce, and every
  algorithm validated on it would be validated on a fiction.

**Not modelled in v1, and named so it cannot be forgotten:** multiple scattering within
participating media, wavelength-dependent BRDF, atmospheric turbulence, detector jitter
beyond a Gaussian IRF, and multi-bounce paths. The last one is what non-line-of-sight
imaging needs, and it is Phase 2 rather than an omission.

### Why the CUDA BVH gets written first, and is not a workaround

The OptiX SDK needs a login this project cannot perform. That would be a blocker except
for one thing: **the measurement needs a baseline anyway.**

Claiming "RT cores are N× faster" requires something to be N× faster *than*, and a
hand-written CUDA BVH traversal is exactly that comparison. So the build order is:

1. CUDA BVH traversal — correct, unaccelerated, and the baseline.
2. OptiX path — same scene, same rays, same output, using the RT cores.
3. The speedup is then measured between two implementations of one interface, on one
   machine, against identical inputs.

Phase 1 is not blocked, and Phase 2 arrives with its comparison already built.

### Validating the simulator itself

This is the part that decides whether anything downstream is worth reading, so it gets
invariants rather than confidence:

- **Analytic case.** A flat plane at known range and incidence has a closed-form return.
  Simulated against analytic must agree to a declared tolerance.
- **Degenerate limit.** As divergence → 0, the return must converge to a scaled copy of
  the transmitted pulse at exactly `2d/c`. If it does not, the geometry term is wrong.
- **Energy.** Total returned energy must fall as 1/d² over a range sweep, and as cos θ
  over an angle sweep, within tolerance. Two independent checks on the radiometry.
- **CPU mirror.** A small scene traced on the host, bit-compared where the arithmetic
  permits and tolerance-compared where it does not.
- **Photon statistics.** With dead time disabled, histogram counts must converge to the
  linear-mode waveform as the pulse count rises. With dead time enabled they must not —
  and the direction of the discrepancy is checkable.

The last one matters: it is the only check that distinguishes "the SPAD model works" from
"the SPAD model is a scaled copy of the other one".

### Language

Per the working rule that a language is chosen by fit:

| | |
|---|---|
| **CUDA / C++** | the simulator. RT cores are reachable only from CUDA-side code, and the histogram accumulation is a GPU problem. |
| **Python** | analysis, plots, dataset packaging. Never in the hot path. |
| **Julia** | reserved for Phase 3, where reconstruction becomes a numerical optimisation problem and Python is too slow to iterate in. |

### Evidence

QuBLAR reuses `rse-hpc-lab`'s harness rather than growing its own, so that a number here
is held to the same standard as a number there. `labkit` enters as a git submodule; the
alternative — copying `evidence.hpp` — guarantees drift, and this project's whole claim is
that its numbers can be trusted.

Declared quantities per measurement: rays traced per second, bytes moved (so G8 reads the
hardware counter), and the invariants above as G6 checks.

## Consequences

**Easier**
- Reconstruction work becomes possible at all, because there is finally an answer to score
  against.
- The RT-core speedup becomes a measured number rather than a claim from a blog post.
- Synthetic datasets can be generated at volume with exact labels.

**Harder**
- Every physics simplification is now a liability that must be written down, because
  downstream results inherit it silently.
- Two implementations of the tracer must be kept in agreement, which is a real maintenance
  cost and is accepted deliberately in exchange for the baseline.

**Open**
- Whether `scenes/` should carry standard scanned geometry or stay procedural. Procedural
  scenes have exactly-known geometry, which is worth more than realism for validation.
- The wavelength to model first. 905 nm and 1550 nm behave differently in atmosphere and
  have different eye-safety limits, which changes achievable power and so the photon rate.

## Action items

**Phase 1 — the generator**
1. [x] Scene representation and BVH build, procedural first
2. [x] CUDA BVH traversal: rays in, hits out
3. [x] Beam model: divergence, Monte Carlo footprint sampling
4. [x] Radiometry: 1/d², cos θ, per-material reflectance
5. [x] Linear-mode waveform accumulation with a Gaussian IRF
6. [x] Ground-truth emission alongside every waveform
7. [x] Invariants: analytic plane, zero-divergence limit, 1/d², cos θ

**Phase 2 — the hardware path**
8. [x] OptiX build once the SDK is present; identical interface
9. [x] Measured speedup against item 2, same scene and rays
10. [ ] Photon-counting detector: Poisson arrivals, dead time, ambient rate
11. [ ] Convergence check against the linear model with dead time off

**Phase 3 — using it**
12. [ ] Reconstruction baselines: peak pick, matched filter, Gaussian decomposition
13. [ ] Scored against truth, not against each other
14. [ ] Multi-bounce tracing, which is the door to non-line-of-sight
