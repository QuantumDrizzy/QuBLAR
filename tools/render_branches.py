"""Render the binary-branch result of check_ising (ADR-007) as a figure.

Tooling over exported files, as ADR-001 allows: reads build/ising_out_*.bin and
.meta, draws with a local N-dimensional renderer (the `ndim` package, unpublished), writes
docs/figures/phase6_branches.png.

Panel 1, the tri-state map in 3D:
    does not exist  every branch says void, an absence the data demand (cyan)
    undecided       branches disagree; drawn as ~~exists~~ (amber)
    exists          all branches say rock; only the pyramid's shell is drawn
                    (grey), or it would hide everything
    truth           the true void's voxels (white, small)
Panels 2 and 3, the x = 0 plane of p_void: data only (lambda = kappa = 0),
then data + prior. Without the prior the deficit streaks along the rays; with
it, the branches settle on a compact void.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ndim.spacetime import Camera, SpacetimeCloud, render_frame, standardise, view_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BG, FG, DIM = "#0d0f14", "#e8e6e1", "#5F5E5A"


def load(build: Path) -> dict:
    meta = dict(line.split(maxsplit=1) for line in (build / "ising_out.meta").read_text().splitlines())
    n = int(meta["nx"])
    shape = (n, n, n)                                   # (z, y, x) in C order
    out = {
        "n": n, "voxel": float(meta["voxel"]),
        "lo": np.array([float(v) for v in meta["lo"].split()]),
        "branches": int(meta["branches"]), "lambda": float(meta["lambda"]),
        "kappa": float(meta["kappa"]),
        "p": np.fromfile(build / "ising_out_pvoid.bin", np.float32).reshape(shape),
        "p0": np.fromfile(build / "ising_out_pvoid_noprior.bin", np.float32).reshape(shape),
        "truth": np.fromfile(build / "ising_out_truth.bin", np.uint8).reshape(shape),
    }
    return out


def centres(d: dict, mask: np.ndarray) -> np.ndarray:
    k, j, i = np.nonzero(mask)
    return np.column_stack([d["lo"][0] + (i + 0.5) * d["voxel"],
                            d["lo"][1] + (j + 0.5) * d["voxel"],
                            d["lo"][2] + (k + 0.5) * d["voxel"]])


def tri_state_panel(d: dict, size: int = 900) -> np.ndarray:
    p = d["p"]
    inside = p >= 0
    shell = inside & ~(np.roll(inside, 1, 0) & np.roll(inside, -1, 0) & np.roll(inside, 1, 1)
                       & np.roll(inside, -1, 1) & np.roll(inside, 1, 2) & np.roll(inside, -1, 2))
    rng = np.random.default_rng(0)
    shell_pts = centres(d, shell)
    shell_pts = shell_pts[rng.random(len(shell_pts)) < 0.25]
    groups = [
        (shell_pts, (0.35, 0.35, 0.33)),
        (centres(d, inside & (p > 0.1) & (p < 0.9)), (0.94, 0.62, 0.15)),
        (centres(d, inside & (p >= 0.9)), (0.25, 0.85, 0.95)),
        (centres(d, d["truth"] > 0) + np.array([0.0, 0.0, 0.0]), (1.0, 1.0, 1.0)),
    ]
    pts = np.vstack([g[0] for g in groups])
    col = np.vstack([np.tile(g[1], (len(g[0]), 1)) for g in groups])
    cloud = SpacetimeCloud(points=pts, colour=col, frame_index=np.zeros(len(pts), int),
                           axis_names=("X", "Y", "Z"), axis_units=("m", "m", "m"),
                           axis_groups=((0, 1, 2),), basis="map")
    std, _, _ = standardise(cloud.points, cloud.axis_groups)
    cam = Camera(azimuth_deg=35.0, elevation_deg=20.0, composite="front", point_size=3,
                 basis="map", extent=2.3)
    return render_frame(std, cloud.colour, view_matrix(3, 3), cam, (size, size))


def plane(d: dict, key: str) -> np.ndarray:
    ix = int(np.floor((0.0 - d["lo"][0]) / d["voxel"]))
    sl = d[key][:, :, ix].copy()                        # (z, y)
    return np.ma.masked_less(sl, 0.0)


def main() -> None:
    build = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build"
    d = load(build)
    fig = plt.figure(figsize=(19, 7), dpi=120)
    fig.patch.set_facecolor(BG)
    ax0 = fig.add_axes([0.0, 0.05, 0.36, 0.85])
    ax0.imshow(tri_state_panel(d))
    ax0.set_axis_off()
    ax0.set_title("tri-state map: cyan does not exist · amber undecided · white truth · grey shell",
                  color=FG, fontsize=10)
    extent = [d["lo"][1], d["lo"][1] + d["n"] * d["voxel"], d["lo"][2], d["lo"][2] + d["n"] * d["voxel"]]
    tz, ty = np.nonzero(d["truth"][:, :, int(np.floor(-d["lo"][0] / d["voxel"]))])
    for pos, key, title in ((0.40, "p0", "x = 0 plane, data only (λ = κ = 0)"),
                            (0.70, "p", f"x = 0 plane, data + prior (λ = {d['lambda']:.1f}, κ = {d['kappa']:.2f})")):
        ax = fig.add_axes([pos, 0.1, 0.27, 0.78])
        im = ax.imshow(plane(d, key), origin="lower", extent=extent, cmap="magma", vmin=0, vmax=1,
                       interpolation="nearest")
        ax.scatter(d["lo"][1] + (ty + 0.5) * d["voxel"], d["lo"][2] + (tz + 0.5) * d["voxel"],
                   s=6, facecolors="none", edgecolors="cyan", linewidths=0.6, label="true void")
        ax.set_xlim(-60, 60)
        ax.set_ylim(30, 120)
        ax.set_title(title, color=FG, fontsize=11)
        ax.set_xlabel("y (m)", color=FG)
        ax.set_ylabel("z (m)", color=FG)
        ax.tick_params(colors=FG, labelsize=8)
        ax.set_facecolor(BG)
        ax.legend(frameon=False, labelcolor=FG, fontsize=8, loc="upper right")
        for s in ax.spines.values():
            s.set_color(DIM)
    cb = fig.colorbar(im, cax=fig.add_axes([0.975, 0.1, 0.008, 0.78]))
    cb.set_label("p(void) over branches", color=FG)
    cb.ax.tick_params(colors=FG, labelsize=8)
    fig.suptitle(f"QuBLAR · muon replica, 3 chambers, {d['branches']} branches · synthetic truth · "
                 "binary branches (ADR-007)", color=FG, fontsize=13)
    out = ROOT / "docs" / "figures" / "phase6_branches.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=BG)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
