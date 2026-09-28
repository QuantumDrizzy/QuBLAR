# ADR-013: Rare-earth solvent-extraction separation sequencing as a QUBO

- Status: pre-registered (frozen copy + SHA-256 in `experiments/ree_cascade/` before any run)
- Date: 2026-09-28
- Scope: synthetic process-design optimisation on published separation factors and a cited ore
  composition. Motivation (REE processing companies) is context only; nothing here says anything
  about any company's process.

## 1. Question

Encode the choice of separation SEQUENCE for a light-REE solvent-extraction plant (which adjacent
split is made on which sub-mixture, in which order) as a QUBO; validate the encoding exactly on tiny
instances; and measure whether QuBLAR's Ising engine (`anneal_branch`, src/ising_recon.hpp, the
engine of the Gset MaxCut bench) finds the proven optimum as reliably and as fast as simulated
annealing (dwave-neal) and better than greedy heuristics.

## 2. Process model (declared simplifications)

- Components ordered by extractability in P507 (HEH/EHP) / HCl: La < Ce < Pr < Nd < "Sm+"
  (Sm + Eu + Gd + heavier lumped, extracted last). A split at s cuts an ordered sub-mixture [a..b]
  into raffinate [a..s] and extract [s+1..b]; only adjacent-key splits (sharp splits).
- Stage count of a split with key pair (s, s+1): Fenske-type minimum stages at total reflux for
  key purity P = 0.999 in both products, N_min = ln[(P/(1-P))^2] / ln(beta_{s+1,s}); operating
  stages N = ceil(2 N_min) (the usual 2 x N_min rule of thumb). Declared: not a full
  Kremser/McCabe-Thiele extraction+scrub design; a Fenske-type count per split, as the brief allows.
- Cost of a split = F_in x N, F_in = REE molar flow entering that split (fraction of the feed).
  Cost of a sequence = sum over its n-1 splits (the classical "flow x stages" size proxy).
  Ce(IV) oxidation/leach pre-separation used at Mountain Pass is NOT modelled (declared).
- Separation factors (primary set "P507-A"):
  beta Ce/La = 8.0 (saponified P507-HCl, industrial value quoted in a Chinese thesis on light-REE
  separation, P507-HCl system; weak source), Pr/Ce = 1.84, Nd/Pr = 1.50 (review table, 1 mol/L P507,
  Sciengine review of acidic phosphorus extractants), Sm/Nd = 10.0 (P507, Gupta & Krishnamurthy,
  Extractive Metallurgy of Rare Earths, 2nd ed., 2015, as cited in later modelling work).
  Secondary set "P507-B": Ce/La 8.0, Pr/Ce 2.2, Nd/Pr 1.6, Sm/Nd 10.0 (same thesis' saponified values).
  Published ranges for context: Ce/La 7.2-14.8, Pr/Ce ~1.2-2.2, Nd/Pr ~1.4-1.6, Sm/Nd 2-12 depending on
  extractant loading and concentration; see also Xie et al., Minerals Engineering 56:10-28 (2014).
- Feed composition (real instance): bastnaesite-barite sövite GH2-57, Mountain Pass, INAA, Haxel,
  USGS Open-File Report 2005-1219, Table 4: La 2.92 %, Ce 5.28 %, Pr 0.39 % (estimated in the source by
  interpolation), Nd 1.15 %, Sm 643, Eu 114, Gd 220, Tb 14.3, Ho 4.34, Tm 0.918, Yb 4.76, Lu 0.673 ug/g.
  Converted to mole fractions (Sm+ lump with the Sm atomic weight).

## 3. QUBO

- Variables x_{a,s,b} = 1 iff sub-mixture [a..b] is split at s (1 <= a <= s < b <= n);
  n(n^2-1)/6 variables (n=5: 20, n=8: 84, n=10: 165, n=12: 286).
- Cost: sum x_{a,s,b} F_[a..b] N(s).
- Constraints (squared penalties, weight P): root (sum_s x_{1,s,n} - 1)^2; for every sub-mixture
  [a..b] with b > a other than the root: (sum_s x_{a,s,b} - sum of its parents)^2, parents =
  x_{a,b,c} (c > b) and x_{c,a-1,b} (c < a).
- Penalty P = 1.1 x (greedy cost). Since all costs are >= 0 and any violated constraint costs >= P,
  every infeasible state has energy >= P > greedy cost >= optimum: the ground state is feasible.
- Engine encoding ("pair rays", no engine change): for each off-diagonal Q_ij a ray with d = 0, w = 1,
  a_i = sqrt|Q_ij|, a_j = sign(Q_ij) sqrt|Q_ij|, which yields Q_ij x_i x_j + |Q_ij|/2 (x_i + x_j); the
  diagonal is corrected through `field`; lambda = 0, no neighbour list. QUBO normalised by P.
- Exact optimum: dynamic programming over (a, b) in O(n^3), cross-checked by enumerating all
  Catalan(n-1) trees. MILP is not needed (the DP is exact and polynomial); declared.

## 4. Solvers (fixed in advance)

- Engine: `anneal_branch` unchanged, Gset MAP schedule (t_hot 5, t_cold 1e-3, 400 anneal + 20 hold
  sweeps), from all-zeros. 64 independent runs per instance (seeds 1..64); "best-of-8" = seeds 1..8.
- neal 0.6.0 SimulatedAnnealingSampler, num_sweeps = 420, 64 reads (default beta range); best-of-8 =
  first 8 reads.
- Greedy: G1 = recursively take the split with the largest beta (fewest stages); G2 = recursively take
  the most flow-balanced split; greedy = the cheaper of G1 and G2.
- Exact: DP (+ enumeration).
- Hit = cost equal to the DP optimum within 1e-9 relative. p_s = hits / 64.
  TTS99 = t_run ln(0.01) / ln(1 - p_s) (t_run if p_s = 1; infinite if p_s = 0); t_run = wall time per
  single anneal/read measured on this PC.

## 5. Instances

- Real: Mountain Pass composition with P507-A (primary) and P507-B (secondary), n = 5.
- Random family: n = 4..12, 10 instances per n (90), compositions ~ Dirichlet(1, ..., 1), adjacent
  betas ~ U[1.4, 2.5], seeded (seed = 1000 n + k).

## 6. Pre-registered rules and win/tie/loss definitions

- **R1 (encoding exactness).** By brute force on n = 4 (one random instance, 2^10 states) and n = 5 (the
  real P507-A instance and one random instance, 2^20 states): the zero-penalty states are exactly the
  Catalan(n-1) valid sequences, and the QUBO minimum equals the DP optimum. Plus: the engine's
  `binary_energy` equals the direct QUBO energy within 1e-6 (normalised units) on 1000 random states
  per instance, all instances. PASS iff all hold.
- **R2 (engine optimality).** Engine best-of-8 reaches the DP optimum on both real instances AND on
  >= 95 % of the random instances, AND its mean relative optimality gap over all instances is <= 1 %.
- **R3 (engine vs neal, per instance).** Win/tie/loss: first by best-of-8 gap (smaller gap wins);
  if equal, by p_s: |p_s,engine - p_s,neal| <= 0.10 is a tie, otherwise the larger p_s wins.
  Overall: "engine beats neal" iff wins > losses with two-sided sign-test p < 0.05 over non-tied
  instances; "neal beats engine" symmetric; otherwise "tie".
- **R4 (time-to-solution).** Median over instances with both p_s > 0 of TTS99(engine)/TTS99(neal):
  <= 0.5 "engine faster", >= 2 "engine slower", otherwise "comparable". Instances where exactly one
  solver has p_s = 0 are counted as a win for the other in a side tally.
- **R5 (vs greedy).** On every instance where greedy is suboptimal, the engine best-of-8 reaches the
  optimum. PASS iff so. The fraction of instances where greedy is suboptimal is reported.
- Expectation written in advance: at <= 286 variables both annealers should mostly reach the optimum;
  the penalty terms dominate the landscape, so p_s per single run may be well below 1 at larger n.
  My prediction: R1 PASS, R2 PASS, R3 tie, R4 comparable. A loss is reported as a loss.

## 7. Visuals

Flowsheet of the optimal sequence for the real composition (with stage counts and flows), energy
landscape of the real instance (all 2^20 states), engine energy-vs-sweep traces (instrumented copy of
the anneal loop, same arithmetic), solver comparison (p_s, gap, TTS vs n).

## 8. What is cut / declared

- Sharp adjacent splits only; no non-sharp/distributed splits, no stage-by-stage cascade simulation,
  no reflux/organic-flow optimisation, no Ce(IV) pre-separation.
- Separation factors are constant per pair (real ones depend on loading, acidity, concentration).
- Time-to-solution is wall time on one CPU thread for the engine and neal's own C loop; both include
  their per-run overheads.

## 9. Results

Filled 2026-09-28 after the run. Frozen copy `experiments/ree_cascade/ADR-013.prereg-frozen.md`, sha256
b8c2d6575059e4952c320b110d729322c43e9ed200e72f675e04383b869510ed (frozen 04:53:25+02:00), unchanged.
Full tables are in `experiments/ree_cascade/RESULTS.md`.

- **R1 FAIL.** The encoding part holds. By brute force on MP_P507-A (2^20), R5_0 (2^20) and R4_0 (2^10),
  the zero-penalty states are exactly the 14 / 14 / 5 Catalan trees, and the QUBO minimum equals the DP
  optimum (58.7304 / 92.6170 / 93.6806). The energy-agreement clause fails: engine `binary_energy` vs
  direct double-precision energy differs by up to 2.18e-5 (normalised) and exceeds 1e-6 on 70/92
  instances. The pair-ray amplitudes are stored as float32 (`BinaryProblem.a`). Hits and gaps are
  recomputed in double, so they are unaffected.
- **R2 FAIL.** Engine best-of-8 is optimal on both real instances but on only 27.8 % of random ones
  (need >= 95 %). The best-of-8 is infeasible on 9 instances (mean gap over all = inf); the mean gap over
  feasible best-of-8 is 12.7 %. neal: real A missed (+10.2 %), real B hit, random 21.1 %, 3 infeasible.
- **R3 tie.** 36 wins / 12 ties / 44 losses, sign test p = 0.43.
- **R4 comparable.** Median TTS99 ratio engine/neal = 1.35 (32 instances with both p_s > 0). Side tally:
  engine-only 11, neal-only 7, neither 42.
- **R5 FAIL.** Greedy is suboptimal on 51/92 (mean 3.3 %, max 23.3 %); the engine best-of-8 is optimal on
  9 of them (neal 8).
- Real feed: optimum 58.73 vs greedy 58.91 (A) and 50.20 vs 50.28 (B). Both have the same optimal
  sequence: La | rest (14 stages), Ce | Pr Nd Sm+ (46 / 36), Pr Nd | Sm+ (12), Pr | Nd (69 / 59).
  Engine p_s 0.17 on both; TTS99 11.7 / 9.5 ms (neal 9.3 / 8.7 ms).
- Scaling: p_s falls from ~0.28 at n=4 to 0 at n >= 10 for both solvers; the feasible fraction falls to
  ~0.1 at n = 12. Engine time per run is ~2x neal's per read.
- Exploratory (10x longer schedules, not pre-registered): R3 tie 37/10/45, TTS ratio 2.08 ("engine
  slower"), random hit8 23 %. No qualitative change.
- Prediction check: R1 PASS / R2 PASS predicted, both FAILED; R3 tie and R4 comparable were predicted
  correctly. Conclusion: the O(n^3) DP is exact and trivially fast. Single-flip annealing on the
  penalty QUBO gets trapped between penalty-separated trees, so the QUBO route is not useful here.
