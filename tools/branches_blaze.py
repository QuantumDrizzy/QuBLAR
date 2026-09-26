"""The ghost bits as a tensor: exact local posterior, compressed with Blaze.

Tooling over check_ising's export build/ising_out_roi.txt (ADR-007 §4):
the conditional QUBO of the 20 voxels most likely to be void, with the rest of
the pyramid fixed at the branch consensus.

  1. Enumerate all 2^20 = 1,048,576 configurations: the exact local posterior
     p(x) ∝ exp(-E(x)). No sampling -- every branch, weighted.
  2. Reshape it to a 20-way tensor, one mode (dimension) per bit, and compress
     it with Blaze (TT/MPS). The bond dimensions say how entangled the ghost
     bits are: rank 1 would mean independent voxels.
  3. Quantize the TT cores to int8 (Blaze phase 8).
  4. Read every bit's marginal p(x_i = 1) straight from the TT, without
     rebuilding the 2^20 tensor, and compare with the exact marginals and with
     the 16-branch estimate from the annealer.

Needs Blaze importable (its python/ directory on sys.path or installed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLAZE = Path(r"C:\Users\Drizzy\Desktop\PR0JECTS\PERSONAL-PR0JECTS\Blaze\python")
if BLAZE.exists():
    sys.path.insert(0, str(BLAZE))
import blaze  # noqa: E402


def load_roi(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    rows = [line.split() for line in lines[1:1 + n]]
    voxel = np.array([int(r[0]) for r in rows])
    p_branch = np.array([float(r[1]) for r in rows])
    truth = np.array([int(r[2]) for r in rows])
    h = np.array([float(r[3]) for r in rows])
    J = np.array([[float(v) for v in line.split()] for line in lines[1 + n:1 + 2 * n]])
    return voxel, p_branch, truth, h, np.triu(J, 1)


def exact_posterior(h: np.ndarray, J: np.ndarray) -> np.ndarray:
    """p over all 2^n configurations; bit i is the i-th most significant."""
    n = len(h)
    idx = np.arange(1 << n, dtype=np.uint32)
    X = ((idx[:, None] >> (n - 1 - np.arange(n))) & 1).astype(np.float64)
    E = X @ h + np.einsum("ki,ij,kj->k", X, J, X, optimize=True)
    p = np.exp(-(E - E.min()))
    return p / p.sum()


def tt_marginals(cores: list[np.ndarray]) -> np.ndarray:
    """p(x_i = 1) from TT cores (r_l, 2, r_r), contracting the other modes by summing."""
    summed = [c.sum(axis=1) for c in cores]                    # (r_l, r_r) each
    n = len(cores)
    left = [np.ones((1,))]
    for k in range(n - 1):
        left.append(left[-1] @ summed[k])
    right = [np.ones((1,))] * n
    acc = np.ones((1,))
    for k in range(n - 1, 0, -1):
        acc = summed[k] @ acc
        right[k - 1] = acc
    z = float(left[-1] @ summed[-1] @ np.ones((1,)))
    return np.array([float(left[k] @ cores[k][:, 1, :] @ right[k]) / z for k in range(n)])


def main() -> None:
    voxel, p_branch, truth, h, J = load_roi(ROOT / "build" / "ising_out_roi.txt")
    n = len(h)
    p = exact_posterior(h, J)
    X_sig = (np.arange(1 << n)[:, None] >> (n - 1 - np.arange(n))) & 1
    exact = (p[:, None] * X_sig).sum(axis=0)
    tensor = p.reshape((2,) * n)
    print(f"region: {n} ghost bits -> {1 << n:,} branches, dense {tensor.nbytes / 2**20:.1f} MiB")
    print(f"  MAP branch probability {p.max():.4f}; states holding 99 % of the mass: "
          f"{int(np.searchsorted(np.cumsum(np.sort(p)[::-1]), 0.99)) + 1:,}")

    report = blaze.analyze_compressibility(tensor, max_rank=64, rel_tol=1e-6)
    print(f"  Blaze diagnostic: {report.get('recommendation', '?')}")
    for tol in (1e-3, 1e-6):
        tt = blaze.compress(tensor, rel_tol=tol)
        rec_err = np.linalg.norm(tt.reconstruct() - tensor) / np.linalg.norm(tensor)
        m = tt_marginals(tt.cores)
        q = blaze.quantize_tt(tt, bits=8).dequantize()
        mq = tt_marginals(q.cores)
        print(f"  TT rel_tol {tol:g}: max bond {max(tt.ranks)}, {tt.nparams():,} params "
              f"({tensor.size / tt.nparams():.0f}x smaller), reconstruction err {rec_err:.1e}")
        print(f"      marginals from the TT (no 2^20 rebuild): max |err| {np.abs(m - exact).max():.1e}; "
              f"int8 cores: {np.abs(mq - exact).max():.1e}")
    print(f"  16-branch annealer vs exact marginals: max |err| {np.abs(p_branch - exact).max():.3f}")
    print("\n   voxel     exact  16-branch  truth")
    for i in np.argsort(-exact):
        print(f"  {voxel[i]:7d}   {exact[i]:.3f}   {p_branch[i]:.3f}      {'void' if truth[i] else 'rock'}")


if __name__ == "__main__":
    main()
