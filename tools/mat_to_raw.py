#!/usr/bin/env python3
"""QuBLAR external-data converter -- ADR-005 item 1.

Converts the released confocal NLOS captures of O'Toole, Lindell & Wetzstein
(Nature 555, 2018; computationalimaging.org) from MATLAB .mat to a flat float32
dump plus a key=value sidecar that src/external.hpp reads. Python is sanctioned
by ADR-001 for dataset packaging, never the hot path.

The .meta is deliberately NOT JSON: parsing JSON in C++ would add a dependency
this repository does not take.

Per-scene parameters are transcribed from the released cnlos_reconstruction.m
(z_offset display crop, diffuse flag, snr, native bin width, z_trim), so the
C++ side never hard-codes facts about someone else's capture.

Usage:
    python tools/mat_to_raw.py data/ext/lct/confocal_nlos_code [scene ...]

Writes data/ext/<scene>.bin and data/ext/<scene>.meta.
"""
import hashlib
import struct
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

# name -> (bin_ps_native, z_trim_native, z_offset_native, isdiffuse, snr, downsample_k)
# Transcribed from cnlos_reconstruction.m (defaults: 4 ps, 600, -, 0, 0.8, 2).
SCENES = {
    "resolution_chart_40cm": (4, 600, 350, 0, 0.8, 2),
    "resolution_chart_65cm": (4, 600, 700, 0, 0.8, 2),
    "dot_chart_40cm":        (4, 600, 350, 0, 0.8, 2),
    "dot_chart_65cm":        (4, 600, 700, 0, 0.8, 2),
    "mannequin":             (4, 600, 300, 0, 0.8, 2),
    "exit_sign":             (4, 600, 600, 0, 0.8, 2),
    "s_u":                   (4, 600, 800, 0, 0.8, 2),
    "outdoor_s":             (4, 600, 700, 0, 0.8, 2),
    "diffuse_s":             (4, 600, 100, 1, 0.08, 2),
    "bunny":                 (8,   0,   0, 1, 0.08, 2),  # simulated by the authors
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(mat_path: Path, out_dir: Path, scene: str) -> None:
    bin_ps, z_trim, z_off, isdiffuse, snr, k = SCENES[scene]
    m = sio.loadmat(str(mat_path))
    rect = np.asarray(m["rect_data"], dtype=np.float32)  # [N, N, M] natural order
    width = float(np.asarray(m["width"]).ravel()[0])
    n, n2, bins = rect.shape
    assert n == n2 and n & (n - 1) == 0 and bins & (bins - 1) == 0, \
        f"{scene}: grid {rect.shape} is not power-of-two (FFT is radix-2)"

    bin_out = out_dir / f"{scene}.bin"
    rect.tofile(bin_out)
    meta = "\n".join([
        f"scene = {scene}",
        f"source_file = {mat_path.name}",
        f"source_sha256 = {sha256(mat_path)}",
        f"n_grid = {n}",
        f"bins_native = {bins}",
        f"bin_ps_native = {bin_ps}",
        f"width_m = {width}",
        f"z_trim_native = {z_trim}",
        f"z_offset_native = {z_off}",
        f"isdiffuse = {isdiffuse}",
        f"snr = {snr}",
        f"downsample_k = {k}",
        f"author_simulated = {1 if scene == 'bunny' else 0}",
        "order = x_y_t",
        "dtype = float32",
    ]) + "\n"
    (out_dir / f"{scene}.meta").write_text(meta)
    print(f"{scene}: {n}x{n}x{bins}, width={width} m, {bin_out.stat().st_size/1e6:.1f} MB")


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/ext/lct/confocal_nlos_code")
    out = src.parents[1]  # data/ext
    names = sys.argv[2:] or [n for n in SCENES if n != "bunny"]
    for scene in names:
        convert(src / f"data_{scene}.mat", out, scene)


if __name__ == "__main__":
    main()
