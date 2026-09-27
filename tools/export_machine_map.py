"""Derive the machine-facing map from the existing p_void export.

Does not re-run annealing and does not retune classify (0.1 / 0.9).

Reads:
  build/ising_out_pvoid.bin   float32, C-order (z, y, x); -1 = outside domain
  build/ising_out.meta

Writes (only what was missing beside p_void):
  build/ising_out_tristate.bin     uint8  0=Exists  1=NotThere  2=Undecided  255=outside
  build/ising_out_traversable.bin  uint8  1 iff Exists (ahead policy); else 0
  build/ising_out_machine.meta     one-page companion keys for loaders

Ahead policy (RESULTS-capabilities.md, check_ahead.cu):
  Exists (p_void <= 0.1)           -> traversable
  Undecided or NotThere            -> not traversable
  outside domain (p_void < 0)      -> not traversable

Self-check: every Undecided voxel has traversable == 0.
Exit non-zero if that invariant fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Frozen with ising_recon.hpp classify(); do not retune.
LO = 0.1
HI = 0.9

EXISTS = 0
NOT_THERE = 1
UNDECIDED = 2
OUTSIDE = 255


def load_meta(path: Path) -> dict[str, str]:
    return dict(line.split(maxsplit=1) for line in path.read_text().splitlines() if line.strip())


def classify_grid(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (tristate uint8, traversable uint8) on the p_void grid."""
    inside = p >= 0.0
    tristate = np.full(p.shape, OUTSIDE, dtype=np.uint8)
    tristate[inside & (p >= HI)] = NOT_THERE
    tristate[inside & (p <= LO)] = EXISTS
    tristate[inside & (p > LO) & (p < HI)] = UNDECIDED
    traversable = (tristate == EXISTS).astype(np.uint8)
    return tristate, traversable


def check_undecided_blocks(tristate: np.ndarray, traversable: np.ndarray) -> None:
    undecided = tristate == UNDECIDED
    n_u = int(undecided.sum())
    leaks = int((undecided & (traversable != 0)).sum())
    if leaks != 0:
        raise SystemExit(
            f"FAIL: undecided must block; {leaks}/{n_u} Undecided voxels have traversable=1"
        )
    print(f"  undecided blocks                            PASS  n_undecided = {n_u}, leaks = 0")


def main() -> int:
    ap = argparse.ArgumentParser(description="Export tristate + traversable from ising_out_pvoid.bin")
    ap.add_argument("--build", type=Path, default=ROOT / "build")
    ap.add_argument("--check-only", action="store_true", help="verify existing masks; do not write")
    args = ap.parse_args()
    build: Path = args.build

    meta_path = build / "ising_out.meta"
    pvoid_path = build / "ising_out_pvoid.bin"
    if not meta_path.is_file() or not pvoid_path.is_file():
        print("need build/ising_out.meta and build/ising_out_pvoid.bin (run check_ising first)",
              file=sys.stderr)
        return 1

    meta = load_meta(meta_path)
    n = int(meta["nx"])
    shape = (n, n, n)
    p = np.fromfile(pvoid_path, np.float32)
    if p.size != n * n * n:
        print(f"pvoid size {p.size} != {n}^3", file=sys.stderr)
        return 1
    p = p.reshape(shape)

    tristate, traversable = classify_grid(p)

    print("QuBLAR -- export_machine_map (from ising_out_pvoid.bin)")
    print(f"  classify lo={LO} hi={HI}; policy: Exists traversable; Undecided|NotThere block")
    print(f"  grid {n}^3; inside = {(p >= 0).sum()}; "
          f"Exists={(tristate == EXISTS).sum()} "
          f"NotThere={(tristate == NOT_THERE).sum()} "
          f"Undecided={(tristate == UNDECIDED).sum()}")

    check_undecided_blocks(tristate, traversable)

    # Synthetic control: force one Undecided cell and confirm the policy.
    p_syn = p.copy()
    # Pick an inside rock cell if any, else first cell.
    inside_idx = np.flatnonzero(p.ravel() >= 0.0)
    if inside_idx.size:
        i = int(inside_idx[0])
        p_syn.ravel()[i] = 0.5  # Undecided under frozen thresholds
        t2, trav2 = classify_grid(p_syn)
        if t2.ravel()[i] != UNDECIDED or trav2.ravel()[i] != 0:
            print("  synthetic undecided blocks                 FAIL", file=sys.stderr)
            return 1
        print("  synthetic undecided blocks                 PASS  p=0.5 -> Undecided, traversable=0")

    if args.check_only:
        t_path = build / "ising_out_tristate.bin"
        trav_path = build / "ising_out_traversable.bin"
        if not t_path.is_file() or not trav_path.is_file():
            print("  --check-only but masks missing; run without --check-only first", file=sys.stderr)
            return 1
        t_disk = np.fromfile(t_path, np.uint8).reshape(shape)
        trav_disk = np.fromfile(trav_path, np.uint8).reshape(shape)
        if not np.array_equal(t_disk, tristate) or not np.array_equal(trav_disk, traversable):
            print("  on-disk masks mismatch recompute           FAIL", file=sys.stderr)
            return 1
        print("  on-disk masks match recompute              PASS")
        return 0

    (build / "ising_out_tristate.bin").write_bytes(tristate.tobytes())
    (build / "ising_out_traversable.bin").write_bytes(traversable.tobytes())
    machine_meta = (
        f"nx {n}\n"
        f"voxel {meta['voxel']}\n"
        f"lo {meta['lo']}\n"
        f"order zyx_C\n"
        f"pvoid ising_out_pvoid.bin float32\n"
        f"tristate ising_out_tristate.bin uint8 0=Exists 1=NotThere 2=Undecided 255=outside\n"
        f"traversable ising_out_traversable.bin uint8 1=Exists_only\n"
        f"classify_lo {LO}\n"
        f"classify_hi {HI}\n"
        f"policy Exists=traversable; Undecided|NotThere|outside=block\n"
        f"note does_not_include_evidence_budget; see check_ising / RESULTS\n"
    )
    (build / "ising_out_machine.meta").write_text(machine_meta)
    print("  wrote ising_out_tristate.bin, ising_out_traversable.bin, ising_out_machine.meta")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
