# ADR-009 — Diffractive layers: the "photonic" in the engine, learned from truth

**Status:** Proposed. A note to design from after 2026-10-05; nothing here is built.
**Date:** 2026-09-27
**Depends on:** ADR-005 (phasor field), ADR-007 (branches), ADR-008 (the contract)

## The picture that prompted it

Two drawings, one above the other:
- **Young's double slit**: a wave through apertures, and interference on the screen.
- **A diffractive deep neural network (D²NN)**: Lin et al., *All-optical machine learning
  using diffractive deep neural networks*, Science 361, 2018. Every point of a diffractive
  layer is a Huygens source that reaches every point of the next layer, so a stack of layers
  is a network whose weights are wave propagation.

## What QuBLAR already has

- `src/phasor.hpp` is the phasor field of Liu et al. (Nature 569, 2019). It treats the relay
  wall as a virtual aperture and propagates with the Rayleigh–Sommerfeld kernel through the
  angular spectrum. **That propagator is one diffractive layer.**
- **The phasor field does not work yet**, and says so: it is XFAIL, with errors of 65.8 cm
  and 54.9 cm, 0 of 2 scenes (RESULTS-phase4).
- Every probe emits its ground truth (ADR-001), so every simulated scene is a labelled
  example.

## Proposal

A diffractive stack inside QuBLAR's inference layer. Each layer is a propagation (the
existing propagator) followed by a learnable phase mask. The masks are trained on synthetic
scenes with known truth, and evaluated on held-out scenes and on the real Nature-2018
captures. That is the literal sense of "it learns from data it never sees": the hidden scene
is never observed directly, only through its truth labels on simulated twins.

## The refusals (the rigour, as in MTLB ADR-0002 §6)

1. **Fix the propagator first.** A learned layer on top of a broken propagator would learn to
   hide the bug. The phasor XFAIL closes, or is explained, before any mask is trained.
2. **A learned layer that does not beat physics is not used.** On held-out scenes it must beat
   LCT and backprojection, and it is scored against truth on the same metrics (detection,
   false alarms, bias, RMSE). A layer that ties is not used.
3. **Amplitudes and probabilities stay apart.** The double slit is exactly where the two
   differ. The wave layers add **amplitudes**, and interference is real there. QuBLAR's
   branches (ADR-007) add **probabilities**: classical samples, with Everett's vocabulary but
   no interference term. Diffraction feeds the forward model, and the Ising inference stays
   classical. Nothing may call a branch ensemble "interference".

## Where it sits in the stack

- **Motor (QuBLAR):** diffractive layers join the forward model next to the ray model:
  waves for photons, straight rays for muons.
- **Compiler (LYTH):** a layer is a 2D FFT, a pointwise multiply and an inverse FFT, all
  memory-bound, which is LYTH's native ground.
- **QGPU (MTLB ADR-0002):** the layer is a QGPU candidate kernel, written workload-up.
- **Compressor (Blaze):** learned masks and propagated fields are tensors, and Blaze's verdict
  says whether they have structure worth compressing.

## Not claimed

No optical hardware, no accuracy number, and no advantage over LCT. The only measured fact
here is that the phasor field currently fails.
