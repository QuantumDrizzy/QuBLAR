# ADR-013 summary (REE SX sequencing as QUBO)

- Frozen 2026-09-28T04:53:25+02:00, sha256 b8c2d657...b869510ed. 92 instances (2 real Mountain Pass feeds
  + 90 random, n = 4..12, 10..286 binary variables).
- The QUBO encoding is correct: on the three brute-forced instances, the feasible states are exactly the
  Catalan trees and the QUBO minimum equals the DP optimum. R1 still FAILS formally because the engine's
  float32 energies differ from double precision by up to 2.2e-5 (tolerance 1e-6).
- R2 FAIL: engine best-of-8 is optimal on both real instances but on only 28 % of random ones, with 9
  infeasible results. R5 FAIL: when greedy is suboptimal (51/92), the engine fixes it only 9 times.
- R3 tie with neal (36 / 12 / 44, p = 0.43). R4 comparable (median TTS ratio 1.35). A 10x longer
  schedule does not change the picture (the TTS ratio becomes 2.08, "engine slower").
- Real feed: optimum 58.73 vs greedy 58.91 (P507-A); 50.20 vs 50.28 (P507-B). Sequence La | Ce | (Pr Nd | Sm+) | Pr/Nd.
- Bottom line: the exact O(n^3) DP beats both annealers outright. Treating this as a QUBO adds nothing.
  Both single-flip annealers get trapped in penalty-separated trees from n >= 7.
