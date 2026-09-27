# Gset MaxCut, same schedule, five graphs

The annealer is the one in `src/ising_recon.hpp`. The schedule is the MAP
schedule already used by `check_ising`'s oracle, copied before these graphs
were scored and not changed after:

`t_hot = 5`, `t_cold = 1e-3`, 400 anneal sweeps, 20 hold sweeps, seeds 1..8,
best cut kept. `lambda = -1`, so the energy of a configuration is minus its
cut. A single edge is cut exactly (`energy = -1`) before any graph is read.
Every edge in G1–G5 has weight +1. A run whose energy is not minus its cut
exits 2.

`dwave-neal` 0.6.0, `num_reads = 8`, `num_sweeps = 420`, `seed = 1`. Its beta
range is the library default. The cut is counted from the assignment, not
taken from the sampler's energy.

Best-known values are the targets the Toshiba SBM benchmark posts for these
graphs (Matsuda, *Benchmarking the MAX-CUT problem on the Simulated
Bifurcation Machine*). They are previous-literature targets, not a cut
measured here on SQBM+.

| graph | engine | seed | neal | best-known target |
|---|---:|---:|---:|---:|
| G1 | 11613 | 6 | 11620 | 11624 |
| G2 | 11617 | 2 | 11597 | 11620 |
| G3 | 11621 | 1 | 11613 | 11622 |
| G4 | 11646 | 1 | 11643 | 11646 |
| G5 | 11616 | 3 | 11631 | 11631 |

G4 meets the published target. G2 and G3 beat neal and miss the target by 3
and 1. G1 and G5 lose to neal. The schedule was not lengthened.

Engine wall time on this machine, one process, about 0.9–1.0 s per graph.
Neal, about 0.2–0.3 s. SHA-256 of the engine assignment (800 bits, `0`/`1`):

| graph | sha256 |
|---|---|
| G1 | `6fa9d0c7e36c590d3d8622e6f548489db688194981ca1f1c8f759b0dd1500a98` |
| G2 | `f59384c0a7495284c74e70c5569c57071db06024322df7f1d456962f9730beab` |
| G3 | `20616526f28f11564a20e13cf7a71d6039c3b5ecd64213a520b3dd107a01b567` |
| G4 | `55156760958f81d0f2f1b5a1d601363f03e38cb38c5e36ca362d7800f5fe5649` |
| G5 | `936ddbc2c0167d4bf22d02147aef02c236072bddb60018aeaf24c2f4149ca5a7` |

This hash is not an ML-DSA signature. Graphs: `data/gset/`. Binary:
`tools/gset_bench.cu`.
