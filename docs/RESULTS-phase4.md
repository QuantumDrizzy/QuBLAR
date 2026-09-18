# External validation: real confocal data through three reconstructions

Measured 2026-09-18. RTX 5060 Ti, CUDA 13.0, Windows toolchain.
Data: O'Toole, Lindell & Wetzstein, *Confocal Non-Line-of-Sight Imaging Based
on the Light-Cone Transform*, Nature 555 (2018) -- captured scenes converted by
`tools/mat_to_raw.py`.

## The FFT

| check | result |
|---|---|
| vs direct DFT, n=16 | rel err 3.8e-15 |
| vs direct DFT, n=256 | rel err 5.9e-14 |
| Parseval, n=1024 | rel err 1.2e-14 |
| ifft(fft(x)) == x, n=1024 | max abs err 4.0e-14 |
| cosine lands exactly two peaks | leakage 2.3e-15 |
| fft3 vs direct 3D DFT, 4^3 | rel err 1.2e-15 |

## The synthetic replica

64x64 relays on a 0.7 m wall patch, two hidden Lambertian patches behind it
(A: side 0.10 m at (−0.10, +0.05, 0.35); B: side 0.08 m at (+0.12, −0.08, 0.60)
in the rectified frame), 4096 bins of 4 ps rectified per relay, downsampled to
16 ps, 1.2 ns direct gate. Voxel grid matched: ratio 1.73 (bar: 4).

| method | object 1 err | object 2 err | precision | boxes detected |
|---|---|---|---|---|
| backprojection | **0.43 cm** | **0.58 cm** | 0.288 | 2/2 |
| LCT (ported) | **0.72 cm** | **1.07 cm** | 0.184 | 2/2 |
| phasor field | 65.8 cm (XFAIL) | 54.9 cm (XFAIL) | 0.002 | 0/2 |

Empty-room control: with-object peak 0.348 vs empty-room peak **exactly 0** --
a reconstruction that answered confidently in an empty room would be
reconstructing its own gate.

## The released captures

Peaks within the authors' own display window (their crop, computed from the
released per-scene parameters -- not tuned here):

| scene | backprojection | LCT | phasor | BP vs LCT |
|---|---|---|---|---|
| diffuse_s | (+0.027, +0.060, 0.513) | (−0.027, +0.060, 0.505) | (+0.115, +0.323, 0.700) | 5.5 cm |
| s_u | (+0.060, +0.180, 0.600) | (−0.049, +0.191, 0.630) | (+0.060, −0.301, 1.180) | 11.4 cm |
| mannequin | (−0.005, −0.038, 0.523) | (−0.016, −0.104, 0.526) | (+0.071, +0.301, 0.815) | 6.7 cm |

The numpy mirror of the authors' MATLAB puts the diffuse_s "S" at
(x=+0.028, y=+0.061, z=0.505); backprojection -- written here from a different
transport model with no shared code -- lands 1 cm away. Two methods that share
nothing but the data agree to centimetres on measurements made by someone
else's hardware years ago.

## The golden check

The C++ LCT port against the numpy mirror of the same MATLAB, full
[512, 64, 64] volume: **max relative error 6.7e-08**. The two ports agree to
float32 quantisation; what remains is rounding order, not structure.

## What the numbers may say (per ADR-005 §7)

- The golden agreement validates the port and the pipeline end to end: loading,
  units, gating, FFT, resampling, radiometric scaling.
- The replica errors validate the three reconstructions against the transport
  model -- the thing only this simulator can score.
- The real-data agreement of BP and LCT validates the loaders and the rectified
  convention against external hardware.
- The phasor field's gap between methods localises unmodelled physics or an
  implementation gap but does not attribute it -- see below.

## What is not yet working: the phasor field

The phasor reconstruction (Liu et al., Nature 569, 2019) is implemented,
band-parameterised, and runs end to end, but does not yet localise the replica
objects (XFAIL in `check_external`, strict: an unexpected pass fails the suite
and forces this section to be rewritten). Its real-data peaks are plausible but
not consistent with BP/LCT. Open hypotheses, in order of suspicion: the
1/(i·omega) depth-averaging sign relative to the e^{+i omega t} synthesis
convention; the per-band normalisation flattening relative phases the original
relies on; aperture padding interacting with the 64x64 wall sampling. The
transport model it inverts is already validated by the LCT result, so the gap
is in the reconstruction, not the data.

## Bugs produced and fixed along the way

Each passed some check at the time:

- **Untrimmed meta keys.** "scene = x" split at '=' leaves the key "scene "
  with a trailing space; every key missed, the file parsed as empty, and the
  error message ("malformed meta") described the symptom, not the cause.
- **Relay plane left at simulator height.** The replica's relays sat at z = 1
  while backprojection searched the rectified frame centred at z = 0; every
  object reappeared at exactly 1 − z, at the correct (x, y), which is precisely
  as confusing as it sounds.
- **A PSF circshift without its z dimension.** Indexing the shift loops by
  (iy, ix) alone rolls slice zero and leaves 1023 slices as an unnormalised
  binary shell: total volume energy preserved to 1.5%, structure uncorrelated
  at −0.08. Energy conservation is not a correctness check.
- **A padded transform filled with an unpadded stride.** tdata's spatial block
  written at stride n into a 2n-strided grid folds it 2:1 against how the FFT
  reads it. Stage-sum checks passed, because a sum is permutation-invariant --
  the check that verified this stage was exactly the check that could not see
  this bug. Both PSF bugs were found by dumping the PSF and diffing, not by
  reasoning about the volume.
- **A zlib stream without its Adler-32 trailer**, caught by the compiler as an
  "unused function" warning -- the writer was never called because the trailer
  assembly had been forgotten.

ADR-001 items 1-14 and ADR-005 items 1-8 are complete (item 8's phasor portion
carries the XFAIL status above).
