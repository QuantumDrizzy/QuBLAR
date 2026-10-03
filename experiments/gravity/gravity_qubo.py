#!/usr/bin/env python3
"""ADR-022: the gravity probe -- quantum underground and underwater mapping as a QUBO.

A 4x4 layer of 50 m voxels (centres 250 m deep), 6x6 surface sensors at 3 uGal station noise
(the sourced Exail AQG class). Bits = "this voxel holds the anomaly at the scene's density
contrast" (ore +1500 kg/m3, or cavity -1600 -- one signed scene per run, 16 bits, ExactSolver
against neal as in the ree discipline). The tri-state map (exists / does not exist / cannot be
decided) comes from the annealed-sample marginals at the 0.9/0.1 margins.

The truth emits as exact spheres; the kernel is point-mass per voxel -- the mismatch is
measured by sub-cube refinement, not assumed away.

    python experiments/gravity/gravity_qubo.py
"""

import json
import math
import pathlib

import numpy as np
import dimod
import neal

G = 6.674e-11
CELL = 50.0
V_CELL = CELL ** 3
Z_LAYER = 250.0
SIGMA_UGAL = 3.0
LAMBDA = math.log(16.0)      # prior: expect one anomaly in the 16 cells
UGAL = 1e-8
SEED = 12345
CX = CY = 4
SX = SY = 6
SUB = 5


def sensor_grid():
    xs = np.linspace(25.0, 225.0, SX)
    ys = np.linspace(25.0, 225.0, SY)
    return [(x, y) for y in ys for x in xs]


def voxel_centres():
    xs = np.linspace(25.0, 175.0, CX)
    ys = np.linspace(25.0, 175.0, CY)
    return [(x, y) for y in ys for x in xs]


def idx_of(cell):
    cx, cy = cell
    return cy * CX + cx


def kernel(drho, radius):
    """A[i, j]: vertical pull (uGal) at sensor i of the scene's sphere (radius, drho) sitting
    at voxel j's centre. Newton's shell theorem: outside a sphere the field IS a point mass
    with the sphere's volume -- exact, not an approximation. The voxel only fixes WHERE."""
    sensors = sensor_grid()
    cells = voxel_centres()
    A = np.zeros((len(sensors), len(cells)))
    for i, (sx, sy) in enumerate(sensors):
        for j, (cx, cy) in enumerate(cells):
            dz = Z_LAYER
            r2 = (sx - cx) ** 2 + (sy - cy) ** 2 + dz ** 2
            gz = G * drho * (4.0 / 3.0 * math.pi * radius ** 3) * dz / r2 ** 1.5
            A[i, j] = gz / UGAL
    return A


def sphere_g_at(radius_m, drho, z_centre, sx, sy, cx, cy):
    """Exact sphere pull (uGal) at a surface point (point-mass far field of the sphere:
    exact for dx = 0, first order off-axis)."""
    vol = 4.0 / 3.0 * math.pi * radius_m ** 3
    dx2 = (sx - cx) ** 2 + (sy - cy) ** 2
    r2 = dx2 + z_centre ** 2
    return G * drho * vol * z_centre / r2 ** 1.5 / UGAL


def cube_refinement(drho, cx, cy, sensor):
    """The voxel's true cube pull by sub-cube refinement (SUB^3 sub-cubes, point-mass each)."""
    sx, sy = sensor
    step = CELL / SUB
    total = 0.0
    for i in range(SUB):
        for jj in range(SUB):
            for k in range(SUB):
                px = cx - CELL / 2 + step * (i + 0.5)
                py = cy - CELL / 2 + step * (jj + 0.5)
                pz = Z_LAYER - CELL / 2 + step * (k + 0.5)
                dx2 = (sx - px) ** 2 + (sy - py) ** 2
                r2 = dx2 + pz ** 2
                total += G * drho * step ** 3 * pz / r2 ** 1.5
    return total / UGAL


SCENES = {
    "ore": {"cell": (1, 2), "radius": 25.0, "drho": +1500.0,
            "note": "ore body: the easy one, ~3.5 sigma"},
    "cavity": {"cell": (2, 0), "radius": 20.0, "drho": -1600.0,
               "note": "cavity: near two sigma after stacking"},
    "cavity_small": {"cell": (0, 3), "radius": 10.0, "drho": -1600.0,
                     "note": "10 m cavity: 0.7 uGal, under the noise -- must land undecided"},
}


def run_scene(name, scene):
    rng = np.random.default_rng(SEED)
    A = kernel(scene["drho"], scene["radius"])
    cx, cy = voxel_centres()[idx_of(scene["cell"])]
    sensors = sensor_grid()
    d = np.array([sphere_g_at(scene["radius"], scene["drho"], Z_LAYER, sx, sy, cx, cy)
                  for sx, sy in sensors])
    d_noisy = d + rng.normal(0.0, SIGMA_UGAL, len(d))

    w = A / SIGMA_UGAL
    y = d_noisy / SIGMA_UGAL
    Qaa = w.T @ w
    lin = -2.0 * (w.T @ y) + LAMBDA   # ||wx-y||^2 expands with -2 y^T w x
    bqm = dimod.BinaryQuadraticModel(
        {j: float(lin[j] + Qaa[j, j]) for j in range(A.shape[1])},
        {(i, j): float(2 * Qaa[i, j]) for i in range(A.shape[1])
         for j in range(i + 1, A.shape[1])},
        float(y @ y), dimod.BINARY)

    ex = dimod.ExactSolver().sample(bqm)
    # dimod does not guarantee record order: take the true minimum explicitly.
    k_opt = int(np.argmin(ex.record.energy))
    energy_opt = float(ex.record.energy[k_opt])
    x_opt = ex.record.sample[k_opt]

    sampler = neal.SimulatedAnnealingSampler()
    ss = sampler.sample(bqm, num_reads=400, num_sweeps=2000, seed=SEED)
    hit = bool(np.any((ss.record.sample == x_opt).all(axis=1)))

    # The engine's convention (ADR-008 / render_dimensions): the tri-state comes from the
    # TEMPERED posterior p_T(x) ~ exp(-E/T) at T = 1, computed EXACTLY over the 16-bit space.
    # neal's quenched endpoints are the annealer-performance check, not the posterior.
    energies = ex.record.energy
    states_bits = ex.record.sample
    w_t = np.exp(-(energies - energies.min()) / 1.0)
    Z_t = w_t.sum()
    p = (states_bits.astype(float) * w_t[:, None]).sum(axis=0) / Z_t

    states = []
    for j, pj in enumerate(p):
        state = ("exists" if pj >= 0.9 else
                 "does not exist" if pj <= 0.1 else "cannot be decided")
        states.append({"voxel": j, "p": round(float(pj), 3), "state": state,
                       "truth": int(j == idx_of(scene["cell"]))})

    committed_wrong = [s for s in states if s["state"] == "exists" and s["truth"] == 0]
    missed = [s for s in states if s["truth"] == 1 and s["state"] == "does not exist"]
    undecided = [s for s in states if s["state"] == "cannot be decided"]

    j_above = idx_of((1, 1))
    point = A[j_above, j_above]
    refined = cube_refinement(scene["drho"], *voxel_centres()[j_above], sensors[j_above])
    g1_err = abs(point - refined) / abs(refined) * 100

    print(f"=== scene: {name} -- {scene['note']} ===")
    print(f"kernel 36x16 at drho {scene['drho']:+.0f} kg/m3; exact optimum {energy_opt:.4f}; "
          f"neal hit: {hit} (400 reads)")
    print(f"truth voxel {idx_of(scene['cell'])} holds the anomaly; its sphere reads "
          f"{sphere_g_at(scene['radius'], scene['drho'], Z_LAYER, cx, cy, cx, cy):.1f} uGal above centre")
    for s in states:
        if s["state"] != "does not exist" or s["truth"]:
            print(f"  voxel {s['voxel']} (truth {s['truth']}): p={s['p']:.3f} -> {s['state']}")
    print(f"committed-wrong: {len(committed_wrong)}; missed: {len(missed)}; "
          f"cannot-be-decided: {len(undecided)}")
    print(f"G1 kernel check: point {point:.2f} vs refined cube {refined:.2f} uGal -> "
          f"err {g1_err:.2f} %\n")

    return {"scene": name, "energy_opt": energy_opt, "hit": hit,
            "states": states, "committed_wrong": len(committed_wrong),
            "missed": len(missed), "undecided": len(undecided),
            "g1_err_pct": round(g1_err, 2)}


def main():
    results = {name: run_scene(name, scene) for name, scene in SCENES.items()}
    path = pathlib.Path(__file__).resolve().parents[2] / "data" / "derived" / "gravity_probe.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
