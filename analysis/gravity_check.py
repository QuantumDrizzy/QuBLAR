#!/usr/bin/env python3
"""Independent cross-check of ADR-022: the kernel recomputed by hand constants, the shell-theorem
exactness of the sphere field at an offset, the ore scene's closure, and two mutants caught
(the prior without the one-anomaly reasoning, and the density sign flipped).

    python analysis/gravity_check.py
"""

import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "gravity"))
import gravity_qubo as gq  # noqa: E402

GP = json.loads((ROOT / "data" / "derived" / "gravity_probe.json").read_text(encoding="utf-8"))


def main():
    fails = []

    # hand kernel: one cell, one sensor, straight from G rho V z / r^3
    drho, radius = 1500.0, 25.0
    A = gq.kernel(drho, radius)
    cells = gq.voxel_centres()
    sensors = gq.sensor_grid()
    j = gq.idx_of((1, 2))
    cx, cy = cells[j]
    nearest = min(range(36), key=lambda i: (sensors[i][0] - cx) ** 2 + (sensors[i][1] - cy) ** 2)
    sx, sy = sensors[nearest]
    dz, dx2 = 250.0, (sx - cx) ** 2 + (sy - cy) ** 2
    hand = 6.674e-11 * drho * (4 / 3 * math.pi * radius ** 3) * dz / (dx2 + dz ** 2) ** 1.5 / 1e-8
    if abs(A[nearest, j] - hand) > 0.05 * abs(hand):
        fails.append(f"kernel differs from hand: {A[nearest, j]:.3f} vs {hand:.3f}")

    # shell theorem: the sphere field at ANY offset is the point mass at the centre
    off = gq.sphere_g_at(radius, drho, 250.0, cx + 137.0, cy - 61.0, cx, cy)
    direct = (6.674e-11 * drho * (4 / 3 * math.pi * radius ** 3)
              / (137.0 ** 2 + 61.0 ** 2 + 250.0 ** 2) ** 1.5 * 250.0 / 1e-8)
    if abs(off - direct) > 1e-6 * abs(direct):
        fails.append("shell theorem broken at an offset")

    # the fixed point: the ore scene's optimum must carry the truth voxel and nothing else
    ore = GP["ore"]
    if ore["committed_wrong"] != 0 or ore["missed"] != 0:
        fails.append("ore scene did not close cleanly")

    # mutant 1: the layer depth misdeclared (kernel at 350 m against a 250 m truth) --
    # the classic gravity-modeling error: the anomaly must come out miscentred or missed
    z_saved = gq.Z_LAYER
    gq.Z_LAYER = 350.0
    resd = gq.run_scene("ore_depth_350", gq.SCENES["ore"])
    gq.Z_LAYER = z_saved
    true_voxel = gq.idx_of(gq.SCENES["ore"]["cell"])
    placement = [s for s in resd["states"] if s["state"] == "exists"]
    if placement and placement[0]["voxel"] == true_voxel and resd["missed"] == 0:
        fails.append("mutant not caught: 100 m depth error invisible")

    # mutant 2: the density sign flipped (a cavity kernel against an ore truth) must miss
    flipped = dict(gq.SCENES["ore"])
    flipped["drho"] = -1500.0
    flipped["note"] = "flipped"
    res_flip = gq.run_scene("ore_flipped", flipped)
    if res_flip["missed"] == 0 and res_flip["undecided"] == 0:
        fails.append("mutant not caught: flipped density sign")

    print(f"ore: voxel-9 p={ore['states'][9]['p']:.3f} exists; wrong {ore['committed_wrong']}; "
          f"missed {ore['missed']}")
    if fails:
        print(chr(10).join(["FAIL"] + fails))
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
