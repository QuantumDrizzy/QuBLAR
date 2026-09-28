# ADR-016: P2, a GPU annealing engine on the P1 weighted integer format

Status: pre-registered. Sections 1-8 were written and frozen before any P2 kernel code
(frozen copy `experiments/p2_gpu/ADR-016.prereg-frozen.md`, hash in
`experiments/p2_gpu/PREREG_SHA256.txt`). Section 9 (results) is filled in only in this file,
after the runs. The frozen rules are never edited; FAILs are reported as they come out.

## 1. Question

P1 (ADR-015) gave an exact integer Metropolis engine on the host. It showed two things:
16 CPU threads leave most of the machine idle, and the fixed T_hot = 5 is far too cold for
K2000 and for large weights. P2 asks three questions:
(a) Can a GPU engine on the same format run the same algorithm at least 10x faster, counted
in spin updates per second?
(b) Can it stay exact, meaning cut == -E in every run and a CPU/GPU result that is bit-identical?
(c) With the same wall time and a weight-scaled temperature applied identically to CPU and
GPU, how do best cut, gap and TTS99 compare?

## 2. What is built (declared before any code)

- `src/ising_gpu.cuh` (new): the device engine, plus the host glue to the P1 `WeightedGraph`.
  `src/ising_weighted.hpp` and `tools/gset_bench.cu` are NOT modified (P1 stays byte-identical,
  see R5).
- `tools/gpu_anneal.cu` (new tool, built by `build.bat`, listed in check.bat's sm_120 loop):
  - `--check [gset_dir]` is the fast self-check that runs in `check.bat`.
  - `--bench <list> <outdir>` runs the section 4 suite.
- **Algorithm: exactly the P1 engine `argos::wising::anneal`.** It is a sequential sweep
  i = 0..n-1 with the same temperature expression per sweep. The draw is splitmix64 with state
  `seed*0x2545F4914F6CDD1D+1`, and a uniform is drawn only when dE > 0. Acceptance is
  `u < exp(-dE/T)`, and all fields and energies are integers.
- **Parallelism: one warp per replica, many replicas per launch** (replica r uses seed r).
  Inside a replica, the warp examines a window of 32 consecutive spins i0..i0+31 at once:
  - Every lane computes dE from the current fields.
  - Draw indices come from a ballot/popc prefix over the lanes with dE > 0.
  - Each lane decides with the draw it would get in sequential order.
  - The first accepting lane L is flipped, and the draw counter advances by the draws of
    lanes <= L. The window then restarts at i0+L+1; if no lane accepts, it restarts at i0+32.
  Spins before L all had dE > 0 and were rejected on unchanged fields, so this is exactly
  the sequential sweep.
- **RNG choice.** splitmix64 is counter-based: the k-th output is mix(s0 + (k+1)*gamma), a pure
  function of seed and k. That is what lets lanes evaluate future draws speculatively. Philox
  is not used because it would make bit-identity with P1's CPU engine impossible. This is a
  stated deviation from the brief's "Philox" example, not from its "counter-based" requirement.
- **Types:**
  - Fields f_i = sum_j w_ij s_j are stored in the narrowest of int8/int16/int32 that holds
    B = max_i sum_j |w_ij|. B > 2^31-1 is refused.
  - dE and all flip updates are computed in 32-bit integers, which is exact by the bound.
  - Per-replica energy is int64.
- **Storage:**
  - Dense weight matrix (int8 if max|w| <= 127, else int32) when average degree >= n/4 and
    n <= 16384.
  - Otherwise CSR (int32 col, int8/int32 weight).
  - For these instances: K2000 is dense; G1, G22 and G81 are CSR.
- **Replica state:**
  - Shared memory when n*(sizeof(field)+1) <= 16 KB (4 warps per block).
  - Otherwise global memory, replica-major so window reads are coalesced.
  - For these instances: G1, G22 and K2000 use shared memory; G81 uses global.
- **Acceptance test on the device** (dE > 0, draw u = u53 * 2^-53):
  - u53 == 0: exact path.
  - xf = -float(dE) * float(1/T); if xf < -40, reject. Here exp(x) < 2^-53 <= u.
  - y = __expf(xf), uf = float(u53)*2^-53. Accept if uf < 0.999*y; reject if uf > 1.001*y.
  - Otherwise use the exact path: accept iff double(u53)*2^-53 < exp(-double(dE)/T), which is
    the P1 expression.
  - Bound: for |x| <= 40 the relative error of y is below 2e-5 (__expf <= 2+1.173|x| ulp, plus
    <= 3*2^-24 relative in xf), far inside the 1e-3 margin. So the fast path never changes a
    decision. The only possible CPU/GPU difference is a 1-ulp difference between device exp and
    MSVC exp inside the exact path, with probability ~2^-53 per exact-path draw.
- **Temperatures** are computed on the host with the same expression as P1's `anneal` and
  uploaded per sweep (double T, float 1/T).

## 3. Instances (from P1's frozen list; same files and SHA-256)

| instance | n | m | weights | target | file SHA-256 |
|---|---:|---:|---|---:|---|
| G1 | 800 | 19176 | +1 | 11624 | 73bf704d8ffc55ba42260ab4cb659e3dcb6e729be70404d2cf476ba4e46d1665 |
| G22 | 2000 | 19990 | +1 | 13359 | 9baeee06eb147b1c9ca42b43be86592d4e6fc60784a85af9be5b63d1362ef28e |
| G81 | 20000 | 40000 | +-1 | 14060 | 74e69d2f5228774cedbdb86da14debf08023556f1d7693b7346ca13df7594d5a |
| K2000 | 2000 | 1999000 | +-1 | 33337 | 9ed615e5e18726914f12740b7f9bedb6b69676477ed5958fac42c8ab0c252ba7 |

Targets and their sources are as in ADR-015 section 3.

## 4. Temperature formula, schedule, batches (frozen)

- **Weight-scaled temperatures (the same formula for CPU and GPU):**
  - sigma = sqrt( (2/n) * sum_edges w_e^2 ), the RMS over spins of the local-field standard
    deviation at random spins.
  - w_min = min |w_e| over w_e != 0.
  - **T_hot = sigma, T_cold = 0.1 * w_min.** This is P1's geometric form
    T = T_hot*(T_cold/T_hot)^(sweep/(A-1)) for sweep < A, then T_cold for the H hold sweeps.
  - Resulting values: G1 sigma = 6.924, G22 4.471, G81 2.000, K2000 44.710; T_cold = 0.1
    for all four.
- **Sweeps:** A = 4000 anneal + H = 200 hold (P1's B10 length), for every run on both devices.
- **GPU batch:** R_gpu = 16384 replicas, seeds 1..16384, in one launch per instance.
  - Batch wall W_gpu is host steady_clock from just before the state-initialising launch to
    after the device-to-host copy of all assignments and energies plus synchronise.
  - It excludes graph upload, allocation and host verification.
  - Cap: 600 s per batch; exceeding it is reported as a failure of R3 for that instance.
- **CPU baseline:** the P1 engine `argos::wising::anneal`, unmodified, with the Budget above,
  on 16 std::threads.
  - Seeds 1, 2, 3, ... are started in order while pool elapsed time < W_gpu, the matched wall.
    A run started before W_gpu completes.
  - W_cpu is the pool wall, from start to last join. R_cpu is the number of completed runs.
- Order per instance: GPU batch first, then the CPU pool with W = W_gpu. Nothing else runs on
  the machine by design.

## 5. Measures (frozen)

- **Per run, both devices:**
  - cut: host `cut_of` on the edge list;
  - E_tracked: the engine's int64;
  - E_scratch: host `energy_scratch` over the CSR;
  - consistent = (cut == -E_tracked == -E_scratch);
  - success = (cut >= target).
- **Throughput:** U = runs completed * (A+H) * n spin updates, and rate = U / wall
  (W_gpu or W_cpu). Speedup = rate_gpu / rate_cpu per instance.
- **Per device and instance:** best cut, gap % = 100 (best - target)/target, mean cut, and
  p = successes / runs.
- **TTS99, amortised.** Runs execute concurrently (16 on the CPU, 16384 on the GPU), so t is
  the batch wall divided by the runs completed: t = W / R. TTS99 = t*ln(0.01)/ln(1-p),
  equal to t if p >= 0.99 and inf if p = 0; this is P1's formula.
  - The 95% CI is P1's bootstrap (B = 2000, splitmix64 seed 20260928, 2.5/97.5 percentiles),
    resampling (success, t) pairs.
  - P1-style per-run latency is reported alongside, not decisive: the CPU mean per-run wall,
    and for the GPU, W_gpu.
- **CPU/GPU agreement:** for every seed s <= min(R_cpu, R_gpu), the GPU assignment must equal
  the CPU assignment (compared by SHA-256), with equal E.
- **Energy oracle:**
  - 1000 random states per instance: bits from splitmix64 seeded 20260929 + instance index,
    bit = top bit of each draw, spin order 0..n-1.
  - Energy is computed by an independent device kernel (int64 over the edge list) and
    compared with host `cut_of`.

## 6. Pass/fail rules (decisive)

- **R1, exact consistency.** In every run on both devices, cut == -E_tracked == -E_scratch.
  PASS iff 0 inconsistent runs.
- **R2, CPU/GPU exact agreement.** On all 4 instances, every shared seed gives identical
  assignments, and all 4000 oracle states give GPU energy == host energy.
  PASS iff 0 differences.
- **R3, throughput >= 10x.** Speedup >= 10 on each of G1, G22, G81 and K2000, and each GPU
  batch completes within its cap. PASS iff all 4.
- **R4, quality reported and consistent.** For both devices on all 4 instances, the results
  report best cut, gap %, p, TTS99 and CI, with dot decimals in the CSV. GPU best cut must be
  >= CPU best cut on each instance, which follows from R2 whenever R_gpu >= R_cpu.
  PASS iff all are reported and the inequality holds on all 4.
- **R5, green and P1 intact.** All of the following must hold:
  - `check.bat` exits 0;
  - the 59 PASS lines it printed after P1 are all still PASS (count unchanged, no FAIL);
  - every new gpu_anneal self-check line passes;
  - the P1 files listed in section 8 have unchanged SHA-256;
  - the rebuilt `gset_bench data\gset\G<k>` output still matches ADR-015 R1(a)'s five hashes.
  PASS iff all hold.
- **R6, prediction: G22 becomes reachable.** GPU p > 0 on G22 (13359 hit at least once in
  16384 runs). PASS iff it holds.
- **R7, prediction: the scaled schedule closes most of P1's K2000 gap.** The GPU best K2000 cut
  is >= 33170 (gap <= 0.5%; P1's best was 32906, gap 1.29%). PASS iff it holds.
- **R8, prediction: quality per wall time.** On every instance where the GPU has p > 0,
  GPU TTS99 < CPU TTS99 (CPU inf counts as larger). PASS iff it holds on all such instances,
  and it is vacuous if there are none; vacuity is stated.

Eight rules. Nothing is re-run with changed parameters. Exploratory runs, if any, are
labelled as such and change no verdict.

**Self-check lines added to check.bat** (all must PASS; they use data only where present):
- On 3 random K16 instances with weights in [-1000, 1000], 1000 random states each, device
  energy == host energy.
- K16 dense path: GPU == CPU assignment for 8 seeds.
- Synthetic sparse +1 graph (n = 300, m = 3000), CSR with shared state: GPU == CPU for 8 seeds.
- The same graph with global state forced: GPU == CPU for 8 seeds.
- A signed-weight sparse graph (n = 300, m = 3000, weights in [-1000, 1000], so int32 weights
  and int16 fields): GPU == CPU for 8 seeds.
- G1: GPU == CPU for seeds 1..64 at the section 4 temperatures with 400+20 sweeps, and
  cut == -E in every run.

## 7. What is not claimed

- No comparison with other published solvers.
- The speedup is against this repo's own 16-thread CPU engine, on this machine, with the same
  algorithm.
- Best-known values are literature targets, not proven optima.
- sigma and 0.1*w_min are a declared formula, not a tuned schedule. No other schedule is tried
  in this ADR.

## 8. Files

- **New:** `src/ising_gpu.cuh`, `tools/gpu_anneal.cu`, this ADR, and `experiments/p2_gpu/`
  (frozen prereg, hash, CSVs, logs, RESULTS.md).
- **Changed:** `build.bat` (builds gpu_anneal) and `check.bat` (sm_120 loop entry plus self-check).
- **P1 files that must stay byte-identical (SHA-256 recorded before P2 work):**
  - `tools/gset_bench.cu` dfc615558fcf3ebbd2ecf2ccc3bdca3f323458f48b5a1ef664b4c4b06c7c31ed
  - `src/ising_weighted.hpp` 619bf73990620bdc2ec25f1a4e91d0c46e9416afdd7f490e38f46c9f91988db4
  - `docs/adr/ADR-015-p1-bench-harness.md` e6fe2d5b97733de1433cc9e4bcb6f545488b3f4daf403745388de596117927f6
  - `experiments/p1_bench/ADR-015.prereg-frozen.md` d0f95bffd6ef3baef6c1b321984765cd60e6e5a460df91c5b4d3386765f8dd89
  - `experiments/p1_bench/PREREG_SHA256.txt` 842a44a4813f47b161214a0f1a07ab1a16d232d5393a4fced5c13e497bbdb312
  - `experiments/p1_bench/RESULTS.md` a8c8d997b5cc0c37c0b29711b57a78f1b63d13ad0b72c18a0a4e2b2285f0864e
  - `experiments/p1_bench/runs.csv` dbd3c289451297e89033f3cabb9c30e4959af7dc8ba93d4cd2795f24a6db5e60
  - `experiments/p1_bench/summary.csv` 8ea9eb97dc24a5a59ea42a0279e917a698dc411349f02492659e0c7b94abc939
  - `experiments/p1_bench/load.csv` 9564c31aef532262bad3932ab19a4d7a6d8bbb2794652320cbce4dd69d2fb9e2
  - `experiments/p1_bench/instances.txt` 34af866a7b435fc5b8812434a0d17452d27cfec8dbb17fd5bd12d6321f521175
  - `experiments/p1_bench/suite_log.txt` 6d4d96910c61921859020594d952b1b769d4fb04937be6f8ffad1280b962fc65
  - `experiments/p1_bench/check_log.txt` 4888ce8dd678d43a07f11199c1cbb17aefe99b5fc841d0d4fcc174804145eada
  - `experiments/p1_bench/repro_G1-G5.csv` 975bf55077e2b897adaad1fd2a0c15a2e4b1a0c97274764d1649f700bf9c685b
  - `experiments/p1_bench/repro_log.txt` 08bff8b57304893475aea32e48dabf45922be0e4f885c949e336495d1605d716
  - `experiments/p1_bench/legacy_r1a.txt` 2b6027f1521486d117ec95b2cd794a684bcfd5127aa6b3f30d94248db0ede7be
- `build.bat` and `check.bat` were changed by P1 and are changed again here, by appending only.
- Data (not committed): as P1.
