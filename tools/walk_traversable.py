"""Walk a voxel path on the exported traversable mask.

Does not re-anneal. Does not retune classify (0.1 / 0.9). Reads only the
on-disk machine map under build/ (see docs/MACHINE-EXPORT.md).

Default path (declared, derived from the grid — not a screenshot pick):
  start  = centre of the in-domain mask (voxels with p_void >= 0):
           component-wise median of in-domain indices, snapped to the
           nearest in-domain voxel if that index is outside the mask.
  target = voxel of maximum p_void among in-domain (measured MAP).
  step   = sign(target - start) per axis (dz, dy, dx), one voxel at a time.
  line   = axis-aligned → existing straight-step (constant step, count =
           max|Δ| + 1); otherwise a 3-D Bresenham grid walk that visits
           every voxel on the segment and stops at the first
           traversable != 1.

Legacy / outside-stop (explicit CLI, same as the former default):
  --start nx//2 nx//2 0 --step 0 0 1 --count nx
  On the existing export that first voxel is outside (p_void = -1).

Stops at the first voxel where traversable != 1. Blocking is a result
(exit 0) when the Undecided→block invariant holds. Exit 1 only on a
leak, a missing required file, or an out-of-array step that is not
reported as outside. Missing masks: run export_machine_map.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

EXISTS = 0
NOT_THERE = 1
UNDECIDED = 2
OUTSIDE = 255

CLASS_NAME = {
    EXISTS: "Exists",
    NOT_THERE: "NotThere",
    UNDECIDED: "Undecided",
    OUTSIDE: "outside",
}


def load_meta(path: Path) -> dict[str, str]:
    return dict(line.split(maxsplit=1) for line in path.read_text().splitlines() if line.strip())


def class_at(tristate: np.ndarray, iz: int, iy: int, ix: int) -> str:
    n = tristate.shape[0]
    if not (0 <= iz < n and 0 <= iy < n and 0 <= ix < n):
        return "outside"
    return CLASS_NAME.get(int(tristate[iz, iy, ix]), f"raw={int(tristate[iz, iy, ix])}")


def check_undecided_blocks(tristate: np.ndarray, traversable: np.ndarray) -> int:
    """Return leak count. Every Undecided cell must have traversable == 0."""
    undecided = tristate == UNDECIDED
    return int((undecided & (traversable != 0)).sum())


def domain_center_and_map(
    pvoid: np.ndarray,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """In-domain centre and MAP voxel (max p_void). Indices are (iz, iy, ix).

    Centre = component-wise median of in-domain indices (p_void >= 0), snapped
    to the nearest in-domain voxel if the median index is outside the mask.
    """
    dom = pvoid >= 0.0
    if not np.any(dom):
        raise ValueError("no in-domain voxels (p_void >= 0)")
    coords = np.argwhere(dom)  # (N, 3) as z,y,x
    med = tuple(int(np.median(coords[:, i])) for i in range(3))
    n = pvoid.shape[0]
    if all(0 <= c < n for c in med) and dom[med]:
        start = med
    else:
        d2 = ((coords - np.array(med, dtype=np.float64)) ** 2).sum(axis=1)
        start = tuple(int(x) for x in coords[int(d2.argmin())])
    masked = np.where(dom, pvoid, -np.inf)
    target = tuple(int(x) for x in np.unravel_index(int(np.argmax(masked)), pvoid.shape))
    return start, target


def sign_step(start: tuple[int, int, int], target: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(int(np.sign(t - s)) for s, t in zip(start, target))


def is_axis_aligned(start: tuple[int, int, int], target: tuple[int, int, int]) -> bool:
    diffs = [abs(t - s) for s, t in zip(start, target)]
    nonzero = sum(1 for d in diffs if d != 0)
    return nonzero <= 1


def bresenham3(start: tuple[int, int, int], target: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """3-D Bresenham: every voxel on the segment from start to target inclusive."""
    z0, y0, x0 = start
    z1, y1, x1 = target
    dz = abs(z1 - z0)
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sz = 1 if z1 >= z0 else -1
    sy = 1 if y1 >= y0 else -1
    sx = 1 if x1 >= x0 else -1
    points: list[tuple[int, int, int]] = []
    z, y, x = z0, y0, x0

    if dx >= dy and dx >= dz:
        ey, ez = 2 * dy - dx, 2 * dz - dx
        for i in range(dx + 1):
            points.append((z, y, x))
            if i == dx:
                break
            if ey >= 0:
                y += sy
                ey -= 2 * dx
            if ez >= 0:
                z += sz
                ez -= 2 * dx
            ey += 2 * dy
            ez += 2 * dz
            x += sx
    elif dy >= dx and dy >= dz:
        ex, ez = 2 * dx - dy, 2 * dz - dy
        for i in range(dy + 1):
            points.append((z, y, x))
            if i == dy:
                break
            if ex >= 0:
                x += sx
                ex -= 2 * dy
            if ez >= 0:
                z += sz
                ez -= 2 * dy
            ex += 2 * dx
            ez += 2 * dz
            y += sy
    else:
        ex, ey = 2 * dx - dz, 2 * dy - dz
        for i in range(dz + 1):
            points.append((z, y, x))
            if i == dz:
                break
            if ex >= 0:
                x += sx
                ex -= 2 * dz
            if ey >= 0:
                y += sy
                ey -= 2 * dz
            ex += 2 * dx
            ey += 2 * dy
            z += sz
    return points


def walk_voxels(
    traversable: np.ndarray,
    tristate: np.ndarray,
    voxels: list[tuple[int, int, int]],
    pvoid: np.ndarray | None,
) -> dict:
    """Walk an explicit voxel list. Report first block or clear finish."""
    n = traversable.shape[0]
    visited: list[tuple[int, int, int]] = []
    for t, (iz, iy, ix) in enumerate(voxels):
        visited.append((iz, iy, ix))
        if not (0 <= iz < n and 0 <= iy < n and 0 <= ix < n):
            return {
                "status": "block",
                "index": (iz, iy, ix),
                "class": "outside",
                "p_void": None,
                "step_t": t,
                "visited": visited,
                "said_outside": True,
            }
        if traversable[iz, iy, ix] != 1:
            p = None if pvoid is None else float(pvoid[iz, iy, ix])
            return {
                "status": "block",
                "index": (iz, iy, ix),
                "class": class_at(tristate, iz, iy, ix),
                "p_void": p,
                "step_t": t,
                "visited": visited,
                "said_outside": False,
            }
    last = visited[-1] if visited else voxels[0]
    p = None
    if pvoid is not None and all(0 <= c < n for c in last):
        p = float(pvoid[last])
    return {
        "status": "clear",
        "index": last,
        "class": class_at(tristate, *last) if all(0 <= c < n for c in last) else "outside",
        "p_void": p,
        "step_t": max(len(voxels) - 1, 0),
        "visited": visited,
        "said_outside": False,
    }


def walk(
    traversable: np.ndarray,
    tristate: np.ndarray,
    start: tuple[int, int, int],
    step: tuple[int, int, int],
    count: int,
    pvoid: np.ndarray | None,
) -> dict:
    """Walk up to `count` voxels with a constant step. Report first block or clear finish."""
    voxels = [
        (start[0] + t * step[0], start[1] + t * step[1], start[2] + t * step[2])
        for t in range(count)
    ]
    return walk_voxels(traversable, tristate, voxels, pvoid)


def maybe_write_slice(
    out: Path,
    pvoid: np.ndarray | None,
    start: tuple[int, int, int],
    step: tuple[int, int, int],
    result: dict,
) -> Path | None:
    """Optional mid-y slice of the measured p_void field (not a camera frame)."""
    if pvoid is None:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  slice skipped (matplotlib not available)")
        return None

    n = pvoid.shape[0]
    iy = start[1]
    # Measured field: p_void on the constant-y plane of the walk start.
    plane = pvoid[:, iy, :].copy()
    display = np.ma.masked_where(plane < 0.0, plane)

    fig, ax = plt.subplots(figsize=(7.0, 6.2), facecolor="#0d0f14")
    ax.set_facecolor("#0d0f14")
    im = ax.imshow(
        display,
        origin="lower",
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
        interpolation="nearest",
    )
    # Full path projected to (ix, iz); on-plane voxels emphasized.
    visited = result["visited"]
    path_x = [v[2] for v in visited if 0 <= v[0] < n and 0 <= v[2] < n]
    path_z = [v[0] for v in visited if 0 <= v[0] < n and 0 <= v[2] < n]
    on_plane_x = [v[2] for v in visited if v[1] == iy and 0 <= v[0] < n and 0 <= v[2] < n]
    on_plane_z = [v[0] for v in visited if v[1] == iy and 0 <= v[0] < n and 0 <= v[2] < n]
    if path_x:
        ax.plot(path_x, path_z, color="#e8e6e1", lw=0.8, alpha=0.55, label="path (proj.)")
    if on_plane_x:
        ax.plot(on_plane_x, on_plane_z, color="#e8e6e1", lw=1.4, alpha=0.95, label="path on slice")
    # Start marker
    if 0 <= start[0] < n and 0 <= start[2] < n:
        ax.scatter(
            [start[2]],
            [start[0]],
            c="#7ec8e3",
            s=40,
            zorder=5,
            marker="o",
            label="start",
        )
    bi = result["index"]
    if result["status"] == "block" and 0 <= bi[0] < n and 0 <= bi[2] < n:
        ax.scatter([bi[2]], [bi[0]], c="#ff6b4a", s=36, zorder=6, marker="x", label="block")
    ax.set_xlabel("ix (x)", color="#e8e6e1")
    ax.set_ylabel("iz (z)", color="#e8e6e1")
    ax.tick_params(colors="#5F5E5A")
    for spine in ax.spines.values():
        spine.set_color("#5F5E5A")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("p_void (measured)", color="#e8e6e1")
    cbar.ax.yaxis.set_tick_params(color="#5F5E5A")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#5F5E5A")
    ax.set_title(
        f"measured p_void · y-slice iy={iy} · step={step}\n"
        "exposure/budget: not in these bins (masks classify; "
        "see check_ising / RESULTS for whether the evidence budget paid)",
        color="#e8e6e1",
        fontsize=9,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sample ising_out_traversable.bin along a voxel path toward measured MAP"
    )
    ap.add_argument("--build", type=Path, default=ROOT / "build")
    ap.add_argument(
        "--start",
        type=int,
        nargs=3,
        metavar=("IZ", "IY", "IX"),
        help="start voxel (z,y,x); default = centre of in-domain (p_void>=0)",
    )
    ap.add_argument(
        "--step",
        type=int,
        nargs=3,
        metavar=("DZ", "DY", "DX"),
        default=None,
        help="constant step (dz,dy,dx); if omitted with --start, uses sign toward MAP "
        "or legacy requires --step. Default path uses Bresenham toward MAP.",
    )
    ap.add_argument(
        "--count",
        type=int,
        default=None,
        help="voxels to walk for constant-step mode; default length of declared path",
    )
    ap.add_argument(
        "--face",
        action="store_true",
        help="legacy outside-stop: start (nx//2,nx//2,0) step (0,0,1) count nx",
    )
    ap.add_argument(
        "--slice",
        type=Path,
        nargs="?",
        const=ROOT / "docs" / "figures" / "walk_traversable_slice.png",
        default=None,
        help="optional: write measured p_void y-slice PNG (default path if flag alone)",
    )
    args = ap.parse_args()
    build: Path = args.build

    meta_path = build / "ising_out.meta"
    tri_path = build / "ising_out_tristate.bin"
    trav_path = build / "ising_out_traversable.bin"
    pvoid_path = build / "ising_out_pvoid.bin"

    if not meta_path.is_file():
        print(f"missing {meta_path}", file=sys.stderr)
        return 1
    if not tri_path.is_file() or not trav_path.is_file():
        print(
            "missing on-disk masks; run: python tools\\export_machine_map.py",
            file=sys.stderr,
        )
        return 1

    meta = load_meta(meta_path)
    n = int(meta["nx"])
    shape = (n, n, n)
    need = n * n * n

    tristate = np.fromfile(tri_path, np.uint8)
    traversable = np.fromfile(trav_path, np.uint8)
    if tristate.size != need or traversable.size != need:
        print(f"mask size mismatch vs nx={n}", file=sys.stderr)
        return 1
    tristate = tristate.reshape(shape)
    traversable = traversable.reshape(shape)

    pvoid = None
    if pvoid_path.is_file():
        p = np.fromfile(pvoid_path, np.float32)
        if p.size == need:
            pvoid = p.reshape(shape)
        else:
            print(f"  warning: pvoid size {p.size} != {n}^3; ignoring p_void")

    # Legacy face start (old default): explicit and reproducible.
    if args.face:
        if args.start is not None or args.step is not None:
            print("--face is mutually exclusive with --start/--step", file=sys.stderr)
            return 1
        start = (n // 2, n // 2, 0)
        step = (0, 0, 1)
        count = int(args.count) if args.count is not None else n
        mode = "constant"
        target = None
    elif args.start is not None and args.step is not None:
        start = tuple(args.start)
        step = tuple(args.step)
        count = int(args.count) if args.count is not None else n
        mode = "constant"
        target = None
    else:
        # Default: domain centre → MAP along Bresenham / axis-aligned step.
        if pvoid is None:
            print(
                "default path needs ising_out_pvoid.bin; run: python tools\\export_machine_map.py",
                file=sys.stderr,
            )
            return 1
        try:
            start_d, target = domain_center_and_map(pvoid)
        except ValueError as e:
            print(f"FAIL: {e}", file=sys.stderr)
            return 1
        if args.start is not None:
            start = tuple(args.start)
        else:
            start = start_d
        step = sign_step(start, target)
        if step == (0, 0, 0):
            # Already at MAP: single-voxel walk.
            mode = "list"
            voxels = [start]
            count = 1
        elif is_axis_aligned(start, target) and args.step is None:
            # Axis-aligned: existing straight-step.
            mode = "constant"
            count = max(abs(t - s) for s, t in zip(start, target)) + 1
            if args.count is not None:
                count = int(args.count)
        elif args.step is not None:
            step = tuple(args.step)
            mode = "constant"
            count = int(args.count) if args.count is not None else (
                max(abs(t - s) for s, t in zip(start, target)) + 1
            )
        else:
            mode = "list"
            voxels = bresenham3(start, target)
            count = len(voxels)
            if args.count is not None:
                voxels = voxels[: int(args.count)]
                count = len(voxels)

    if mode == "constant" and count < 1:
        print("count must be >= 1", file=sys.stderr)
        return 1

    leaks = check_undecided_blocks(tristate, traversable)
    print("QuBLAR -- walk_traversable")
    print(f"  grid {n}^3 (z,y,x); start={start} step={step} count={count}")
    if target is not None:
        print(f"  target MAP (max p_void)={target}  mode={mode}")
    print(f"  invariant Undecided -> traversable==0: leaks={leaks}")
    if leaks != 0:
        print(f"FAIL: {leaks} Undecided voxels have traversable=1", file=sys.stderr)
        return 1

    if mode == "list":
        result = walk_voxels(traversable, tristate, voxels, pvoid)
    else:
        result = walk(traversable, tristate, start, step, count, pvoid)
    idx = result["index"]
    cls = result["class"]
    p_str = "n/a" if result["p_void"] is None else f"{result['p_void']:.6f}"

    if result["status"] == "block":
        print(f"  BLOCK at index (iz,iy,ix)={idx}  class={cls}  p_void={p_str}  t={result['step_t']}")
    else:
        print(f"  CLEAR through {count} voxels; last (iz,iy,ix)={idx}  class={cls}  p_void={p_str}")

    # Out-of-array without saying so: should not happen if walk reports outside.
    iz, iy, ix = idx
    oob = not (0 <= iz < n and 0 <= iy < n and 0 <= ix < n)
    if oob and not result.get("said_outside"):
        print("FAIL: path left the array without reporting outside", file=sys.stderr)
        return 1

    slice_path = None
    if args.slice is not None:
        slice_path = maybe_write_slice(args.slice, pvoid, start, step, result)
        if slice_path is not None:
            print(f"  wrote slice {slice_path}")

    print("  exit 0 (invariant holds; block is a result)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
