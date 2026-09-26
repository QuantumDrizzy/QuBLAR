"""The tri-state map of check_ising (ADR-007), orbited once.

Same data and colours as panel 1 of render_branches.py; only the camera moves, so what
turns is the view, never the result:
    does not exist  every branch says void (cyan)
    undecided       branches disagree (amber)
    truth           the true void's voxels (white)
    shell           the pyramid's surface, subsampled (grey)

Reads build/ising_out_*.bin and .meta; writes docs/figures/tristate_orbit.gif.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_branches import load, tri_state_panel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    build = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build"
    d = load(build)
    found = int(((d["p"] >= 0.9) & (d["p"] >= 0)).sum())
    frames = []
    for az in np.arange(0.0, 360.0, 5.0):
        rgb = tri_state_panel(d, size=640, azimuth_deg=35.0 + az, extent=3.1)
        img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                              if rgb.dtype != np.uint8 else rgb).convert("RGB")
        ImageDraw.Draw(img).text(
            (12, img.height - 22),
            f"cyan: does not exist ({found} voxels) - amber: undecided - white: truth",
            fill=(150, 150, 145))
        frames.append(img.quantize(colors=64, method=Image.MEDIANCUT))
    out = ROOT / "docs" / "figures" / "tristate_orbit.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=70, loop=0,
                   optimize=True)
    print(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MiB, {len(frames)} frames), "
          f"{found} voxels marked 'does not exist'")


if __name__ == "__main__":
    main()
