# Multi-bounce transport, and locating an object nobody can see

Measured 2026-09-11. RTX 5060 Ti, CUDA 13.0, Windows toolchain.

Sensor at the origin. Relay wall at z = 1 m facing it. A hidden patch between the two,
off to the side, placed so no line from the sensor to any relay point passes through it.
The sensor never illuminates the patch and never collects light from its direction — the
confocal optics see only the wall.

Gate: 4096 bins of 4 ps. System response 100 ps FWHM.

## Two arrivals, both arithmetic

| feature | expected | measured path error |
|---|---|---|
| relay wall | `t = 2\|p − s\|/c` | 0.19 mm |
| hidden patch | `t = 2(\|p − s\| + r)/c` | 10.0 mm, at t = 12.96 ns |

The two differ in amplitude by four orders of magnitude, so a model that reproduced one
and not the other would still look like a plausible transient. The 10 mm figure for the
patch is not error in the model — it is a 6 cm square seen at an angle, and the peak sits
at its weighted centre rather than at the centre of its bounding box.

## The 1/r⁴ that makes NLOS hard

A hidden patch of fixed size, moved away from the relay wall:

| r (m) | energy | E·r⁴ |
|---:|---:|---:|
| 0.300 | 4.465e-02 | 3.616e-04 |
| 0.450 | 8.876e-03 | 3.640e-04 |
| 0.600 | 2.808e-03 | 3.640e-04 |
| 0.750 | 1.159e-03 | 3.667e-04 |

`E·r⁴` constant to **1.4%** across a 2.5× range sweep. The signal drops by a factor of 38
over that span, which is the whole reason non-line-of-sight imaging is difficult and why
it needs single-photon detectors.

The tracer does not hard-code this. It samples solid angle, so it carries `1/r²` per path;
the second factor appears on its own because a patch of fixed size subtends less solid
angle as it recedes. Had the geometry been wrong, this would have come out as `1/r²` — and
no arrival-time check would have revealed it, because the timing would still be exact.

## Seeing around the corner

16 × 16 relay points over a 1 m square of wall, 2²⁰ paths each, first-bounce wall return
included and gated out at reconstruction time the way a real system must.

    hidden object at   (+0.300, -0.200, +0.550)
    reconstructed at   (+0.289, -0.195, +0.554)

**Error 1.25 cm, on a 1.56 cm voxel grid.** Sub-voxel, from nothing but the timing of light
that bounced off a wall.

The control matters as much as the result: with the object removed, the same pipeline
produces a peak of exactly zero. A reconstruction that produces a confident answer from an
empty room is reconstructing its own gate.

## The grid and the pulse must be matched, and getting that wrong is not a blur

The first attempt reconstructed the object **49 cm from where it was**, with the transport
model already fully correct — every arrival time and the entire `1/r⁴` curve above were
passing at the time.

The cause: a voxel is scored by sampling the transient at the arrival time implied by its
*centre*. At 4 ps bins one bin is 0.6 mm of range. A 2.5 cm voxel puts its centre up to
20 bins away from the true arrival while the pulse spans 3. The voxel holding the object
therefore sampled the transient nowhere near the return, no voxel lay on every shell at
once, and the peak went wherever the sidelobes happened to agree.

It is now a stated precondition rather than a lesson:

    ratio = (2 × voxel half-diagonal) / (pulse sigma × c)

checked before the reconstruction runs. The bar is **4**, and it is derived rather than
chosen — the pulse splat is truncated at 4σ, so beyond that a voxel samples the return at
exactly zero and the method cannot work at all. The broken configuration ran at 10. This
one runs at 1.84 and lands inside one voxel.

A 100 ps system response is used rather than 30 ps for the same reason. 30 ps looks better
on paper and makes a centimetre-scale grid unusable; 100 ps is what a SPAD-based NLOS rig
actually has.

## What is not modelled, named so it is not forgotten

- **Four bounces and beyond**, and interreflection inside the hidden volume. Below the
  three-bounce term at these albedos but not zero — the first place to look if a
  reconstruction grows a low-amplitude halo.
- **Non-Lambertian relay walls.** Real walls are not perfectly diffuse and the confocal
  model is sensitive to it.
- **No occluder**, which is a consequence rather than an omission: under the confocal
  assumption the beam lights only the relay point and the detector accepts only light
  arriving from it, so the occluder in a real rig enforces what the optics already enforce
  here. Worth writing down, because "we forgot the occluder" and "the occluder is
  redundant" look identical in the output.

ADR-001 items 1–14 are complete.
