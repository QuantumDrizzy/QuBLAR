# Phase 2 result — CUDA BVH against RT cores

Measured 2026-09-11 on an RTX 5060 Ti (Blackwell, sm_120), RT core version 40, OptiX
9.1.0, CUDA 13.0, Windows host toolchain for both sides. 1048576 rays per launch, median
of 11 timed launches after 3 warm-ups, CUDA events around the launch only.

## What this number is

**The traversal, and nothing else.** Rays in, nearest hits out, identical structs,
identical scene, identical ray buffers. The sensor model is not ported to OptiX and will
not be: OptiX programs cannot use shared memory, and the LiDAR kernel accumulates its
waveform there.

So this is an **upper bound** on what the full simulator could gain, not a prediction of
it. The simulator does per-sample radiometry, pulse splatting and truth accumulation that
no traversal speedup touches. "QuBLAR is N× faster with RT cores" would be a false
sentence.

## Agreement

Ten configurations — five scene sizes, two ray regimes — 1048576 rays each. **Zero
disagreements**: hit and miss identical, distances within 1e-6 m, normals within 1e-6 of
parallel. Both tracers are also watertight on a closed box.

That agreement was not free; getting there found two real defects, below.

One predicted difference did show up and is reported rather than suppressed. ADR-002 said
in advance that a ray striking a shared edge may be attributed to either of the two
triangles, and that both answers are correct as long as the distance matches. The harness
prints that count as "edge ties": 21 of 1048576 coherent rays and 1–4 incoherent ones, with
distances agreeing to 1e-6 m in every case. A prediction that is never checked is a guess,
so the column is always on rather than printed only on failure.

## Throughput

| triangles | rays | baseline Mray/s | OptiX Mray/s | ratio |
|---:|---|---:|---:|---:|
| 432 | coherent | 1917 | 3616 | 1.89 |
| 432 | incoherent | 568 | 3496 | **6.15** |
| 3072 | coherent | 1546 | 3730 | 2.41 |
| 3072 | incoherent | 471 | 2945 | 6.25 |
| 12288 | coherent | 1406 | 3706 | 2.64 |
| 12288 | incoherent | 436 | 2912 | 6.68 |
| 49152 | coherent | 1167 | 3770 | 3.23 |
| 49152 | incoherent | 422 | 2821 | 6.68 |
| 110592 | coherent | 922 | 3884 | 4.21 |
| 110592 | incoherent | 414 | 2685 | 6.49 |

Two things the sweep shows that a single number would have hidden:

**The baseline degrades with scene size and OptiX does not.** From 432 to 110592
triangles the hand-written traversal loses 52% of its coherent throughput (1917 → 922)
while OptiX gains slightly (3616 → 3884). The ratio therefore rises with scene
complexity, and quoting it from any one scene would say more about the scene than the
hardware.

**Ray coherence matters more than scene size.** The baseline loses roughly 3× going from
a 0.05 rad cone to the full sphere; OptiX loses about 20%. Benchmarking only coherent rays
would have understated the gap by a factor of three. A scanning LiDAR fires coherent rays;
the multi-bounce work in Phase 3 will not.

## Build cost, reported apart and never added in

| triangles | host BVH build | OptiX build | host nodes | GAS bytes |
|---:|---|---|---:|---:|
| 432 | — | 0.12 ms | 7136 B | 9856 |
| 110592 | — | 0.59 ms | 1814496 B | 1892480 |

The hardware structure is slightly *smaller* than the 32-byte-node array it replaces, at
110592 triangles: 1.89 MB against 1.81 MB — near parity, which is worth knowing, since the
intuition that a hardware format must be bulkier is wrong here.

Maximum tree depth reached 19 of the traversal's 32-entry stack at the largest scene. The
stack is now reported on every run rather than assumed, because the traversal silently
drops children when it overflows — that is a wrong answer with no error, and it was for a
while the leading suspect for a disagreement it turned out not to cause.

## What the agreement cost, stated because the ratio benefits from it

The baseline was **faster before it was correct**. With Möller–Trumbore and a barycentric
epsilon it ran at 2679 Mray/s on 432 coherent rays; with the watertight intersector it runs
at 1917. Across the sweep the correct version costs 1.4–1.9× in throughput.

OptiX's numbers did not move (3576 → 3616 on the same case). **So part of the speedup in
the table above came from the baseline slowing down, not from the RT cores speeding up.**
Against the leaky baseline the 432-triangle coherent ratio was 1.34; against the correct
one it is 1.89. The correct comparison is the one in the table — an incorrect baseline is
not a baseline — but the reason the number moved is not something to leave for a reader to
discover.

## Two defects found by the comparison

**The scene generator was open.** `make_box` computed a cell's far edge as `x0 + step`
while the neighbouring cell computed the same edge as `-h + (i+1)*step`. In float32 those
differ by about one ULP, roughly 2.4e-7 — a real crack between quads that are supposed to
share an edge. Three of a million rays escaped a closed box at subdiv 96, two missed by
OptiX and one by the baseline. That two-sided split was the diagnosis: OptiX's intersector
is watertight by construction, so a leak striking *both* tracers cannot be in either.

**FMA contraction broke watertightness.** With the generator fixed, one leak remained, and
Woop's intersector fixed it only when compiled with `-fmad=false`. Written as `a*b - c*d`,
nvcc fuses the edge functions into an FMA; the two triangles sharing an edge then no longer
compute exactly opposite values, and a ray can fall outside both. The host build was clean
throughout, because the host does not contract — which is what made it confusing. The fix
is `__fmul_rn`/`__fsub_rn` in six expressions rather than `-fmad=false` across the program,
which would have disabled fusion in the radiometry to solve a problem that does not live
there.

The watertightness test was raised from 30000 rays at subdiv 8 to 1048576 at subdiv 96. The
old one passed throughout both bugs. A test that cannot see the failure it is named after
is decoration.
