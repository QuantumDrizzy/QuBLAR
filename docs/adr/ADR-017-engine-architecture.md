# ADR-017 — QμBLAR as an engine: five layers, from one pipeline to a platform

**Status:** Accepted 2026-09-28. L1 built: see `docs/RESULTS-engine-L1.md`
**Date:** 2026-09-28
**Deciders:** Antonio
**Supersedes, as direction:** the one-pipeline shape of ADR-007/008. Their results stand.

## Context: what QμBLAR is today (level 0)

QμBLAR works, and it is soldered to its first problem:
- `build_binary_problem` takes the muon model and its views directly;
- the unknowns are binary voxels;
- the prior is a pairwise Ising prior (κ, λ);
- inference is one simulated annealer, plus exact enumeration on a small region.

Every new domain (mine, tunnel, Khufu, REE) has been a new check bolted onto that. Gset shows
the annealer is not a faster optimiser: against `dwave-neal` it wins 2, loses 2 and ties the
target on G4, 3–4× slower. The value is not speed. It is **seeing without seeing, with a
certificate**: exists / does not exist / undecided, the evidence in nats, and how much
measurement a decision costs.

To carry that into robots, vehicles, computer vision, mining, KY-RO, SUBSTRATE and AETHER, the
engine needs layers, not more checks.

## Decision: five layers, each behind an interface, each with one refusal

```
  L5  FUSION     several sensors, one posterior               (robots, cars, mining)
  L4  ACTIVE     where to measure next; time                  (active perception, survey design)
  L3  LEARNED    priors learned from twins; quantum-sampled priors (QML, honestly scoped)
  L2  INFERENCE  annealing, parallel tempering, population annealing -> evidence
  L1  MODEL      unknowns x forward operator x prior          (the contract)
  L0  TODAY      muons + binary voxels + Ising prior + SA     (kept as the first instance)
```

### L1 — The model: three plug-ins instead of one pipeline

| Plug-in | Interface | First instances | Next |
|---|---|---|---|
| **Unknowns** | a discrete field on a grid or graph | binary voxels (today) | Potts labels (air / rock / ore / water), bit-planes for continuous values, occupancy + class |
| **Forward operator** | `apply(x)` and `adjoint(r)`, matrix-free on the GPU, with a declared noise model | muon rays (Beer–Lambert), photon transient (LiDAR/NLOS) | camera projection, radar, X-ray, seismic travel time, gravity/magnetics (AETHER) |
| **Prior** | an energy over the unknowns | pairwise Ising (κ, λ) | higher-order and learned (L3) |

**Refusal:** an operator that fails its adjoint test (⟨Ax, r⟩ = ⟨x, Aᵀr⟩ to tolerance) does not load.

### L2 — Inference: more than one engine, and the evidence as a number

- **Simulated annealing** stays as the reference.
- **Parallel tempering** mixes where SA gets stuck (the G1/G5 losses are the test).
- **Population annealing** is GPU-native, thousands of replicas, and estimates the **log-evidence**
  of a model. That turns "does it exist" into a comparison of models (void vs no void, ore vs
  no ore) with a number. It generalises today's evidence budget.
- **Exact regions** stay: enumeration and Blaze TT on small regions of interest.

**Refusal:** a sampler that misses the exact posterior on the 2¹⁶ oracle, or disagrees with SA
on marginals beyond tolerance, does not run.

### L3 — Learned priors, and QML without science fiction

- **Energy-based priors learned from simulated twins with ground truth.** RBM/Boltzmann priors
  first, deep energy models after. The same annealer samples them. This is where deep learning
  enters: as a prior the physics can overrule, never as the answer.
- **Quantum-sampled priors (QML).** A transverse-field (quantum Boltzmann) prior, sampled on the
  simulated QPU (state vector, ~30 qubits), or a q-sample circuit via Blaze's MPS → circuit.
  It is scoped as research. No quantum advantage is claimed; it is benchmarked against the
  classical sampler on the same data.

**Refusal (ADR-009's rule, generalised):** a learned prior that does not beat the pairwise prior
on held-out scenes with truth is not used.

### L4 — Active: where to measure next, and time

- **Expected information gain.** Which next measurement (a detector position, a LiDAR ray, a
  camera pose, exposure days) most reduces the undecided mass. This is survey design for mining
  and active perception for robots, from the same code. GARY's information meter is the
  yardstick.
- **Time.** A temporal prior links frames, so ghost bits persist and move. This is tracking what
  is occluded or around a corner, for vehicles.

**Refusal:** a planner whose chosen measurement does not beat a random one in information gain,
on truth, is not used.

### L5 — Fusion: several sensors, one posterior

Muons + gravity + LiDAR + camera enter as operators on one field, each with its own noise
model. The certificate says which sensor paid for which bit.

**Refusal:** fusion that is not at least as good as the best single sensor, on truth, is not used.

## The ecosystem around it

- **LYTH** compiles the hot kernels: operator apply/adjoint, sweeps, replica updates.
- **Blaze** compresses posteriors and returns them with its verdict.
- **MTLB/QPU** hosts the quantum-sampled priors.
- **Bastion** signs certificates.
- **KY-RO, SUBSTRATE, AETHER** consume the engine through a **C ABI + Python SDK**. KY-RO does
  not get the arc's code; it gets the engine with its own operator and prior.

## The RTX 5060 Ti budget (16 GB)

A 512³ binary field is 16 MB as bits and 512 MB as float marginals. 4096 population-annealing
replicas × 10⁶ variables as bits is 512 MB. The GPU is not the limit at these sizes. Memory
movement is, which is LYTH's job. Every run declares its budget and prints start and done
(`run_talk`), and one heavy job runs at a time.

## Build order: each level closes before the next opens

1. **L1:** extract the three interfaces. The muon pipeline becomes the first instance, and every
   existing check passes unchanged. This is the refactor that removes the ceiling.
2. **L2:** parallel tempering, then population annealing with log-evidence, validated on the
   2¹⁶ oracle and on Gset G1/G5.
3. **L4-lite:** information gain for mining survey design, answering "which detector next, how
   many days".
4. **L3:** a learned RBM prior on a twin set, then the quantum-sampled prior as research.
5. **L5:** fusion, starting with muons + gravity.

## Not claimed

No quantum advantage, no real-world accuracy beyond the scenes with truth, and no speed record.
Every level is a capability with a gate, not a promise.
