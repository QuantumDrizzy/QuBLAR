# ADR-009 — A double-slit neural network: the "photonic" in the engine, learned from truth

**Status:** Proposed. A note to design from after 2026-10-05; nothing here is built.
**Date:** 2026-09-27
**Depends on:** ADR-005 (phasor field), ADR-007 (branches), ADR-008 (the contract)

## One object, not two

The picture that prompted this is a double slit and a neural network, drawn one above the
other. They are **the same object**: a network whose layers *are* slit screens. This is what
a diffractive deep neural network is (Lin et al., *All-optical machine learning using
diffractive deep neural networks*, Science 361, 2018), read the owner's way:

| Neural network | Double slit |
|---|---|
| a layer | a screen of apertures (slits) |
| a weight | an aperture's transmission and phase, **learned** |
| the weighted sum | interference: each aperture is a Huygens source reaching every point of the next screen |
| the connection between layers | free-space propagation (Rayleigh–Sommerfeld) |
| the nonlinearity | measurement: the detector sees intensity, \|field\|² |

Nothing sits beside the interference to compute. **The interference is the computation.**

## What QuBLAR already has

- `src/phasor.hpp` is the phasor field of Liu et al. (Nature 569, 2019). It treats the relay
  wall as a virtual aperture and propagates with the Rayleigh–Sommerfeld kernel through the
  angular spectrum. **That propagator is one diffractive layer.**
- **The phasor field does not work yet**, and says so: it is XFAIL, with errors of 65.8 cm
  and 54.9 cm, 0 of 2 scenes (RESULTS-phase4).
- Every probe emits its ground truth (ADR-001), so every simulated scene is a labelled
  example.

## Proposal

The double-slit network as QuBLAR's photonic layer. Each layer is a learnable slit screen
followed by a propagation, using the existing propagator. The masks are trained on synthetic
scenes with known truth, and evaluated on held-out scenes and on the real Nature-2018
captures. That is the literal sense of "it learns from data it never sees": the hidden scene
is never observed directly, only through its truth labels on simulated twins.

## The refusals (the rigour, as in MTLB ADR-0002 §6)

1. **Fix the propagator first.** A learned layer on top of a broken propagator would learn to
   hide the bug. The phasor XFAIL closes, or is explained, before any mask is trained.
2. **A learned layer that does not beat physics is not used.** On held-out scenes it must beat
   LCT and backprojection, and it is scored against truth on the same metrics (detection,
   false alarms, bias, RMSE). A layer that ties is not used.
3. **Inside the network everything is amplitude, and the handover is a measurement.** The
   double slit is exactly where amplitudes and probabilities differ. The wave layers add **amplitudes**, and interference is real there. QuBLAR's
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
