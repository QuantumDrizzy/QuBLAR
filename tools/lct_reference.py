#!/usr/bin/env python3
"""QuBLAR LCT reference mirror -- ADR-005 item 5, first port check.

A line-by-line numpy mirror of the authors' released cnlos_reconstruction.m,
run on the converted captures. The C++ port in src/lct.hpp must agree with
this output numerically. Two ports written from the same source by the same
author catch transcription slips, not conceptual errors -- that is all this
check is claimed to catch (ADR-005 §5).

Deliberate deviation from the MATLAB, mirrored on both sides: c is
kSpeedOfLight = 299792458 m/s, not their 3e8 (0.07%; using the same constant
in both ports keeps the comparison exact).

Usage:
    python tools/lct_reference.py data/ext <scene> [...]

Writes <scene>_lctvol.bin (float32, cropped display volume, [z,y,x] natural
order), <scene>_front.png (max projection over depth) and prints the peak.
"""
import struct
import sys
import zlib
from math import log2
from pathlib import Path

import numpy as np

C = 299792458.0  # kSpeedOfLight; see header note


def load(scene_dir: Path, scene: str):
    meta = {}
    for line in (scene_dir / f"{scene}.meta").read_text().splitlines():
        k, _, v = line.partition("=")
        meta[k.strip()] = v.strip()
    n = int(meta["n_grid"])
    bins = int(meta["bins_native"])
    rect = np.fromfile(scene_dir / f"{scene}.bin", dtype=np.float32).reshape(n, n, bins)
    return rect, meta


def resampling_operators(m: int):
    """The released sparse t->sqrt(t) resampling, built dense and averaged down.

    mtx is M^2 x M with a 1/sqrt(x) weight at column ceil(sqrt(x)); both it and
    its transpose are then halved log2(M) times to match the time-downsampled
    resolution. Ported exactly, including the 1-based ceil(sqrt(x))."""
    m2 = m * m
    x = np.arange(1, m2 + 1)
    a = np.zeros((m2, m))
    a[x - 1, np.ceil(np.sqrt(x)).astype(int) - 1] = 1.0
    a *= (1.0 / np.sqrt(x))[:, None]
    at = a.T.copy()
    for _ in range(int(round(log2(m)))):
        a = 0.5 * (a[0::2, :] + a[1::2, :])
        at = 0.5 * (at[:, 0::2] + at[:, 1::2])
    return a, at


def lct(rect: np.ndarray, meta: dict, full: bool = False):
    n = rect.shape[0]
    bin_s = float(meta["bin_ps_native"]) * 1e-12
    width = float(meta["width_m"])
    snr = float(meta["snr"])
    isdiffuse = int(meta["isdiffuse"])
    k_down = int(meta["downsample_k"])
    z_trim = int(meta["z_trim_native"])
    z_off = int(meta["z_offset_native"])

    for _ in range(k_down):  # downsample the time axis by 2^K, summing pairs
        rect = rect[:, :, 0::2] + rect[:, :, 1::2]
        bin_s *= 2.0
        z_trim = round(z_trim / 2)
        z_off = round(z_off / 2)
    m = rect.shape[2]
    rect[:, :, :z_trim] = 0.0  # gate the direct return off

    rng = m * C * bin_s
    slope = width / rng

    # --- blur kernel (their definePsf), 1-based quirks preserved ---
    ax = np.linspace(-1.0, 1.0, 2 * n)
    az = np.linspace(0.0, 2.0, 2 * m)
    gz, gy, gx = np.meshgrid(az, ax, ax, indexing="ij")
    psf = np.abs(((4.0 * slope) ** 2) * (gx**2 + gy**2) - gz)
    psf = (psf == psf.min(axis=0, keepdims=True)).astype(np.float64)
    psf /= psf[:, n - 1, n - 1].sum()  # their psf(:,U,U) with U = n (1-based)
    psf /= np.linalg.norm(psf)
    psf = np.roll(psf, (0, n, n), axis=(0, 1, 2))  # their circshift([0 U U])

    fpsf = np.fft.fftn(psf)
    invpsf = np.conj(fpsf) / (np.abs(fpsf) ** 2 + 1.0 / snr)

    mtx, mtxi = resampling_operators(m)

    data = np.transpose(rect, (2, 1, 0)).astype(np.float64)  # their permute([3 2 1])
    gz1 = np.linspace(0.0, 1.0, m)[:, None, None]
    data *= gz1**4 if isdiffuse else gz1**2

    tdata = np.zeros((2 * m, 2 * n, 2 * n), dtype=np.complex128)
    tdata[:m, :n, :n] = (mtx @ data.reshape(m, -1)).reshape(m, n, n)

    tvol = np.fft.ifftn(np.fft.fftn(tdata) * invpsf)[:m, :n, :n]
    vol = np.maximum((mtxi @ tvol.reshape(m, -1)).reshape(m, n, n).real, 0.0)

    # --- their display crop: keep the wall-patch-sized depth window. Skipped
    # for the golden dump: check_external compares the full physical volume.
    if full:
        zs = np.linspace(0.0, rng / 2.0, m)
        xs = np.linspace(-width, width, n)
        return vol.astype(np.float32), zs, xs

    ind = int(round(m * 2.0 * width / (rng / 2.0)))
    vol = vol[::-1 if False else 1, :, ::-1]  # flip x (their dim3 reversal)
    vol = vol[z_off: min(ind + z_off, m)]
    zs = np.linspace(0.0, rng / 2.0, m)[z_off: min(ind + z_off, m)]
    xs = np.linspace(-width, width, n)
    return vol.astype(np.float32), zs, xs


def write_png(path: Path, img: np.ndarray) -> None:
    """8-bit grayscale PNG via zlib -- no imaging dependency for a sanity view."""
    lo, hi = float(img.min()), float(img.max())
    g = np.zeros_like(img) if hi <= lo else (img - lo) / (hi - lo)
    rows = ((1.0 - g) * 255).astype(np.uint8)  # invert: bright object on dark bg
    h, w = rows.shape
    raw = b"".join(b"\x00" + rows[i].tobytes() for i in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9))
           + chunk(b"IEND", b""))
    path.write_bytes(png)


def main() -> None:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/ext")
    args = sys.argv[2:]
    dump_full = "--dump-full" in args
    args = [a for a in args if a != "--dump-full"]
    for scene in args or ["diffuse_s", "s_u"]:
        rect, meta = load(d, scene)
        vol, zs, xs = lct(rect, meta)
        vol.tofile(d / f"{scene}_lctvol.bin")
        front = vol.max(axis=0)  # projection over depth: the "front view"
        write_png(d / f"{scene}_front.png", front)
        iz, iy, ix = np.unravel_index(int(np.argmax(vol)), vol.shape)
        print(f"{scene}: vol {vol.shape}, peak {vol.max():.4g} at "
              f"z={zs[iz]:.3f} m, x={xs[ix]:+.3f} m, y={xs[iy]:+.3f} m")

        if dump_full:
            # the UNcropped, UNflipped full volume [m, n, n] float32, z on the
            # uniform axis -- the golden file the C++ port is checked against
            # by check_external (ADR-005 item 5).
            full, _, _ = lct(rect, meta, full=True)
            full.tofile(d / f"{scene}_lctref.bin")
            print(f"{scene}: golden volume {full.shape} -> {scene}_lctref.bin")


if __name__ == "__main__":
    main()
