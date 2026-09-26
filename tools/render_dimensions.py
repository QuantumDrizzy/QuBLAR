"""Ghost bits as dimensions: the 3-, 6- and 9-cube of QuBLAR's most uncertain voxels.

Reads build/ising_out_roi_uncertain.txt: the exact conditional QUBO of the 12
voxels whose annealed branches disagreed most (check_ising, ADR-007). All
2^12 configurations are enumerated, which is the exact local posterior. The
k most uncertain bits (k = 3, 6, 9) are kept by marginalising the others.

The film is an anneal. The posterior is tempered, p_T(x) ∝ exp(-E(x)/T): hot,
every branch glows; cooled to T = 1, the true posterior, the mass collapses
onto the branch the data support. The temperature is printed on every frame.

Every configuration of k bits is a vertex of the k-cube: a branch. Every
edge is one bit flip, the move the annealer makes. A vertex's size and
brightness are its exact posterior probability, and the most probable branch
is ringed. The cubes turn through their k dimensions (Givens rotations) and
are projected to the page, so what moves is the view, never the data.

Writes docs/figures/dimensions_369.gif. Synthetic pyramid, known truth.
"""

from __future__ import annotations

import io
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BG, FG = "#07080c", "#e8e6e1"


def load(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    rows = [line.split() for line in lines[1:1 + n]]
    h = np.array([float(r[3]) for r in rows])
    J = np.triu(np.array([[float(v) for v in line.split()] for line in lines[1 + n:1 + 2 * n]]), 1)
    truth = np.array([int(r[2]) for r in rows])
    return h, J, truth


def energies(h, J):
    n = len(h)
    X = ((np.arange(1 << n)[:, None] >> (n - 1 - np.arange(n))) & 1).astype(float)
    return X, X @ h + np.einsum("ki,ij,kj->k", X, J, X)


def tempered(E, T):
    p = np.exp(-(E - E.min()) / T)
    return p / p.sum()


def marginal_over(X, p, keep):
    k = len(keep)
    idx = (X[:, keep] * (1 << (k - 1 - np.arange(k)))).sum(axis=1).astype(int)
    return np.bincount(idx, weights=p, minlength=1 << k)


def givens(k, i, j, a):
    r = np.eye(k)
    r[i, i] = r[j, j] = np.cos(a)
    r[i, j], r[j, i] = -np.sin(a), np.sin(a)
    return r


def panel(ax, k, pk, t):
    verts = ((np.arange(1 << k)[:, None] >> (k - 1 - np.arange(k))) & 1) * 2.0 - 1.0
    rot = np.eye(k)
    for n, (i, j) in enumerate(combinations(range(k), 2)):
        rot = rot @ givens(k, i, j, 2 * np.pi * t * (0.35 + 0.13 * (n % 5)) / 3.0)
    petrie = np.array([[np.cos(np.pi * m / k) for m in range(k)],
                       [np.sin(np.pi * m / k) for m in range(k)]])
    xy = (verts @ rot.T) @ petrie.T
    xy /= np.abs(xy).max() * 1.08
    edges = [(a, a ^ (1 << b)) for a in range(1 << k) for b in range(k) if a < a ^ (1 << b)]
    w = np.array([np.sqrt(pk[a] * pk[b]) for a, b in edges])
    w = w / w.max()
    ax.add_collection(LineCollection([xy[[a, b]] for a, b in edges],
                                     colors=plt.cm.cool(0.15 + 0.85 * w),
                                     linewidths=0.3 + 1.6 * w, alpha=0.25 + 0.75 * w))
    s = pk / pk.max()
    order = np.argsort(s)
    ax.scatter(xy[order, 0], xy[order, 1], s=6 + 260 * s[order], c=s[order], cmap="magma",
               vmin=0, vmax=1, edgecolors="none", zorder=3)
    m = int(np.argmax(pk))
    ax.scatter(*xy[m], s=420, facecolors="none", edgecolors="#40d9f2", linewidths=1.6, zorder=4)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{k} ghost bits · {1 << k} branches", color=FG, fontsize=13, pad=4)


def main() -> None:
    h, J, truth = load(ROOT / "build" / "ising_out_roi_uncertain.txt")
    X, E = energies(h, J)
    p1 = tempered(E, 1.0)
    marg = (p1[:, None] * X).sum(axis=0)
    correct = int(((marg > 0.5).astype(int) == truth).sum())
    # the bits to show: the ones the ANNEALED branches were least sure of (order in file
    # is by voxel; exact marginals are all ~0/1, so rank by sensitivity to temperature)
    p_warm = tempered(E, 4.0)
    order = list(np.argsort(np.abs((p_warm[:, None] * X).sum(axis=0) - 0.5)))
    keeps = {k: sorted(order[:k]) for k in (3, 6, 9)}
    t_hot = float(np.ptp(E)) / 2.0
    hold, cool, rest = 18, 60, 22
    temps = [t_hot] * hold + list(t_hot * (1.0 / t_hot) ** (np.arange(cool) / (cool - 1))) + [1.0] * rest
    frames = []
    for f, T in enumerate(temps):
        pT = tempered(E, T)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), dpi=76)
        fig.patch.set_facecolor(BG)
        for ax, k in zip(axes, (3, 6, 9)):
            ax.set_facecolor(BG)
            panel(ax, k, marginal_over(X, pT, keeps[k]), f / len(temps))
        fig.text(0.5, 0.93, f"annealing · T = {T:6.2f}", color="#40d9f2", ha="center",
                 fontsize=14, family="monospace")
        fig.text(0.5, 0.035, "QuBLAR · Ising photonic engine · tempered exact posterior of the most "
                 "uncertain hidden voxels (synthetic pyramid, known truth) · vertex = branch · "
                 f"edge = one bit flip · at T = 1: {correct}/{len(truth)} bits match the truth",
                 color="#9a988f", ha="center", fontsize=9)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.08, wspace=0.02)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=BG)
        plt.close(fig)
        frames.append(Image.open(buf).convert("RGB").quantize(colors=112, method=Image.MEDIANCUT))
    out = ROOT / "docs" / "figures" / "dimensions_369.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=65, loop=0, optimize=True)
    print(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MiB), T {t_hot:.1f} -> 1, "
          f"{correct}/{len(truth)} bits correct at T = 1")


if __name__ == "__main__":
    main()
