# ADR-008 — QuBLAR as an Ising photonic engine: the contract

**Status:** accepted
**Date:** 2026-09-26
**Follows:** ADR-007 (binary branches). This fixes what QuBLAR *is*, so that the labs using
it (SUBSTRATE first) depend on a contract rather than on a check file.

## Context

Phases 1–6 built two things that turn out to be one engine:

- **Probes with truth.** Photons (LiDAR, NLOS) and muons are sent through a scene. The
  arrivals are counted, and the truth is emitted beside every measurement.
- **Inference of what is not there.** Every unknown is a bit; the data and a declared prior
  form a QUBO; annealed branches give a tri-state map (exists / does not exist /
  undecided) with measured certainty; Blaze compresses the ghost bits' tensor.

The owner's name for the whole is an **Ising photonic engine**. "Photonic" names its
native probes; the muon mode shows that the inference half is probe-agnostic. "Ising" names
the reconstruction. The engine is quantum-inspired only: Everett's many weighted worlds are
the branch ensemble, without the ontology. There are no qubits. The solver is classical
annealing.

## Decision

### 1. Three layers, one direction of data

```
  PROBE                     FORWARD MODEL                 INFERENCE
  photons | muons | …  →    config x → predicted data  →  QUBO: data misfit + declared prior
  truth emitted                A (sparse, per ray)          branches (annealed samples)
                                                            tri-state map + certainty
                                                            exact ROI posterior → Blaze TT
```

A lab can enter at any layer:
- with **its own measurements and forward model**, for example SUBSTRATE with GeoPulse data;
- with **QuBLAR's probes**, for synthetic ground truth.

### 2. The contract, what goes in and what comes out

- **In:** unknowns and their domain; rays or measurement rows with (d_b, w_b) and their
  coefficients a_bv; the declared priors (λ, κ or p₀) and the solver schedule.
- **Out:**
  - p(1) per unknown and the branches;
  - the tri-state labels, with thresholds stated;
  - the data-driven or prior-driven label (from a prior-off rerun);
  - the evidence budget (the data's nats against the prior's);
  - optionally an ROI's exact QUBO and its Blaze TT.
- **Invariants, each a check:**
  - the annealer agrees with exhaustive enumeration on a small instance;
  - an empty control produces no confident "does not exist";
  - every certainty the prior bought is labelled as such.

### 3. Solvers are pluggable, the problem is not

The QUBO is the interface. Today it is solved by host simulated annealing, with freezing of
provable variables. The same problem can later go to:
- **OSCILLON**, a simulated photonic, coherent Ising machine;
- **TRELLIS → D-Wave**, a real quantum annealer, when access exists.

Every new solver must match the host oracle on shared instances before its numbers are
reported.

### 4. General engine, with SUBSTRATE as its first consumer

QuBLAR stays its own repository and stays general. SUBSTRATE, the quantum lab, is where it
is used most, and a text bridge (ghost bits of a text) remains possible later. The bridges
depend on the contract (§2) through exported files (`build/ising_out_*`) and, later, a C++
library target. They never depend on check programs.

### 5. Efficiency is part of correctness

The same answer for less compute:
- freeze what is provable;
- enumerate small regions exactly instead of sampling them;
- compress what is sharp.

Shorter schedules and fewer branches were measured to lose voxels (RESULTS-phase6) and are
not defaults. The engine uses at most half the host threads, leaving the owner's machine
headroom.

## Consequences

- The README leads with "Ising photonic engine", quantum-inspired, no qubits.
- The next QuBLAR phase is the first real-data bridge: SUBSTRATE and GeoPulse feeding the
  contract of §2 (SUBSTRATE ADR-0001).
