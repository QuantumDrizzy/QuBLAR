# ADR-004 — Multi-bounce transport, and seeing around a corner

**Status:** accepted
**Date:** 2026-09-11
**Implements:** ADR-001 action item 14.

## Context

Everything so far assumes light leaves the sensor, strikes one surface, and comes back.
That assumption is baked into the radiometry (`w·ρ·cosθ/d²`), into the truth emission (one
entry per surface the footprint struck), and into every reconstruction baseline (a return
is a surface at `ct/2`).

Non-line-of-sight imaging breaks all three. The light that carries information about a
hidden object never travels in a straight line to it: it scatters off a relay wall, strikes
the object, scatters back to the wall, and only then returns. The range equation is
different, the truth is no longer "what the footprint hit", and `ct/2` is no longer a
range — it is a sum of two unknown segments.

This is why item 14 was kept out of ADR-003: it changes the transport model, not the
instrument.

## Decision

### 1. Confocal three-bounce, and that is why there is no occluder

The geometry is the standard confocal one: the beam is aimed at a point `p` on a relay
wall, and the detector collects only from that same point `p`. The path is

    sensor → p → q → p → sensor,     total length  L = 2(|p − s| + |q − p|)

Real NLOS rigs place an occluder so the detector cannot see the hidden object directly.
**This simulator models no occluder, and that is a consequence rather than an omission:**
under the confocal assumption the beam illuminates only `p`, so the hidden object is never
directly lit, and the detector accepts only light arriving from `p`, so a direct return
from the object cannot be collected either. The occluder in a real rig enforces what the
confocal optics already enforce here. Writing it down matters because "we forgot the
occluder" and "the occluder is redundant under these assumptions" look identical in the
output.

### 2. The weight is derived, not copied from a paper

Sampling is cosine-weighted over the hemisphere at `p`. For one sampled path:

- the first bounce contributes `f_p·cosθ_p / pdf = (ρ_w/π · cosθ_p)/(cosθ_p/π) = ρ_w`,
  so the sampling cosine cancels exactly;
- `q` is Lambertian, so it returns `ρ_q/π · cosθ_q` toward `p`;
- the receiver at `p` subtends `A_p cosθ_p / r²`.

        weight  ∝  ρ_w · ρ_q · cosθ_q · cosθ_p / r²

**This is not a contradiction of the 1/r⁴ that the NLOS literature quotes.** That figure is
per unit *area* of hidden surface, and converting solid angle to area brings a second
`cosθ_q/r²`. Both statements are the same model; they differ in what is held fixed. The
simulator samples solid angle, so it carries `1/r²` per path and produces `1/r⁴` per unit
area on its own — and the falloff check below measures the `1/r⁴`, because that is the
thing that makes NLOS hard.

### 3. The direct wall return stays in the transient

The first-bounce return off the relay wall itself is orders of magnitude stronger than
anything from the hidden object. It could be suppressed, and is not, for two reasons: it is
present in real data and must be gated out by whoever processes it, and it gives the
validation a second, independent arrival whose time is exactly `2|p − s|/c`.

So the transient carries two features with closed-form arrival times, at very different
amplitudes. A simulator that gets one and not the other is wrong in a way a single peak
would not reveal.

### 4. Reconstruction is filtered backprojection, and it is scored on position

For a confocal system the set of hidden points consistent with an arrival at time `t` from
relay point `p` is the **sphere** `|q − p| = ct/2 − |p − s|`. Backprojection votes each
measurement onto that sphere across a voxel grid; the votes agree only where the object is.

That is the oldest NLOS reconstruction and deliberately so — it makes no assumption about
the object beyond the transport model, which is exactly what a baseline should do. Light
cone transform and f–k migration are faster and sharper, and both are *inversions of this
same forward model*; a baseline that shares assumptions with them would not be a check on
those assumptions.

Scoring is the distance from the reconstructed peak to the known hidden object, in metres.
Not a correlation, not an IoU: this is a localisation task, and a metric that can be good
while the position is wrong would defeat the purpose.

### 5. What is not modelled, named so it cannot be forgotten

- **Four bounces and beyond.** Their contribution is well below the three-bounce term for
  the albedos here, but it is not zero, and this is the first place to look if a
  reconstruction shows a low-amplitude halo.
- **Interreflection within the hidden volume.** Same reasoning.
- **Wall retroreflection and non-Lambertian relay surfaces.** Real walls are not perfectly
  diffuse, and the confocal model is sensitive to that.
- **Ambient light in the transient**, which the detector layer adds separately.

## Consequences

- The pulse splat is extracted from `lidar_trace` into a shared device function. Two
  copies of an energy-conserving bin integral is exactly the duplication that let the
  baseline and the physics drift apart in Phase 2.
- Truth for NLOS is the hidden object's position, not a per-beam return list. The existing
  `TruthReturn` machinery does not apply and is not forced to.
- The transient is a photon budget like any other, so the SPAD layer from ADR-003 applies
  unchanged — pile-up on a three-bounce signal is a real and severe problem, and it is now
  expressible.

## Action items

1. [x] Extract the pulse splat so both transport models share one implementation
2. [x] Confocal three-bounce path tracer with cosine-weighted sampling
3. [x] Check: the direct wall return arrives at exactly `2|p − s|/c`
4. [x] Check: the three-bounce return arrives at exactly `2(|p − s| + r)/c`
5. [x] Check: hidden-object signal falls as `1/r⁴` per unit area
6. [x] Confocal backprojection onto a voxel grid
7. [x] Score: distance from the reconstructed peak to the known object

Measured outcome: `docs/RESULTS-phase3b.md`.
