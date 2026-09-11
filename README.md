# QuBLAR

A LiDAR simulator that emits the **truth alongside every measurement**, so reconstruction
algorithms can be scored against what actually happened rather than against each other.

Most range data is a point cloud: one distance per beam, with whatever produced it already
thrown away. A real beam has angular divergence, so its footprint at range covers an area
that can contain several surfaces at different distances — and the interesting physics
lives exactly where it straddles an edge. This simulator records the full time-resolved
return, and next to it the list of surfaces that made it.

Built on an RTX 5060 Ti (Blackwell, sm_120) with CUDA 13.0 and OptiX 9.1.0.

---

## What it does

| | |
|---|---|
| **Geometry** | A hand-written CUDA BVH, and the same queries through the RT cores via OptiX. The two agree on 1048576 rays across five scene sizes and two coherence regimes. |
| **Sensor** | Beam divergence with stratified solid-angle sampling, the extended-target range equation derived rather than copied, and a bin-integrated Gaussian pulse. |
| **Truth** | Per beam: the distinct surfaces the footprint struck, with range, incidence and the fraction of beam energy each returned. |
| **Detector** | Single-photon (SPAD/TCSPC): inhomogeneous Poisson arrivals by inverting the cumulative rate, non-paralysable dead time, ambient background. Produces real pile-up. |
| **Reconstruction** | Peak pick, matched filter, greedy deconvolution, plus Coates pile-up correction — each scored against truth on detection, false alarms, bias and RMSE, never combined into one number. |
| **Multi-bounce** | Confocal three-bounce transport, and non-line-of-sight reconstruction by filtered backprojection. |

## Some measured results

- **RT cores run traversal 1.9–6.7× faster** than the CUDA baseline, rising with scene
  complexity and much larger for incoherent rays than coherent ones
  ([details](docs/RESULTS-phase2.md) — including the part where the baseline got *slower*
  when it was made correct, which is where some of that ratio came from).
- **Pile-up costs 3.3 cm of range, systematically short.** Coates correction removes
  essentially all of it ([details](docs/RESULTS-phase3.md)).
- **An object nobody can see is located to 1.25 cm** from nothing but the timing of light
  that bounced off a wall ([details](docs/RESULTS-phase3b.md)).

## Build and check

Windows only, and deliberately: OptiX does not work under WSL on this machine
(`libnvoptix.so.1` there is a 14 KB stub), and building the baseline and the accelerated
path with different compilers would put a confound inside the one number they exist to
produce.

```bat
build.bat        REM vcvars64, then nvcc for all five binaries
check.bat        REM every invariant; non-zero exit if any fails
sanitize.bat     REM compute-sanitizer memcheck and racecheck
```

`check.bat` verifies with `cuobjdump` that every binary really contains `sm_120` code
before reporting a single number, because `-arch=sm_120` is a request and not a
confirmation.

## How to read this repository

Design first. Each phase has a decision record written *before* the code, and a results
document written after:

| | |
|---|---|
| [ADR-001](docs/adr/ADR-001-photon-transport-lidar.md) | the physics, what is deliberately not modelled, and why the CUDA BVH was written before the OptiX one |
| [ADR-002](docs/adr/ADR-002-optix-path.md) | the RT-core path, and **what its speedup is allowed to claim** |
| [ADR-003](docs/adr/ADR-003-detector-and-reconstruction.md) | the photon-counting detector, and scoring reconstruction against truth |
| [ADR-004](docs/adr/ADR-004-multibounce-nlos.md) | multi-bounce transport and seeing around a corner |

## On trusting the numbers

Every claim here is checked against a closed form, a statistical law, or an independent
implementation — not against a tolerance that could have been tuned until a test passed.
Where a threshold exists it is derived and the derivation is in the source.

The bugs found along the way are recorded next to the code that had them, because each one
passed a test suite at the time:

- an empty AABB collapsed the BVH to a single node, and the brute-force agreement test
  still passed, because a one-leaf hierarchy *is* brute force;
- an axis-aligned ray made the slab test compute `0 * inf = NaN`, and every comparison
  against NaN is false, so rays sailed through solid geometry;
- the mesh generator computed a shared edge two different ways, one ULP apart, opening a
  2.4e-7 m crack that took a million rays to find;
- FMA contraction broke the watertight intersector on the device while the host build,
  from identical source, stayed perfect;
- a missing `__syncthreads()` let one thread rescale the photon budget another was still
  reading — the histogram stayed entirely plausible;
- a reconstruction grid ten times coarser than the range resolution put a hidden object
  49 cm from where it was, with the transport model already completely correct.

ADR-001's Evidence section carries an amendment recording that its original plan — reusing
an external harness as a submodule — is **not** what was built, along with what is
genuinely missing as a result (no hardware-counter coverage, no physics ceiling for the
ray-throughput figures). It is amended rather than rewritten on purpose.
