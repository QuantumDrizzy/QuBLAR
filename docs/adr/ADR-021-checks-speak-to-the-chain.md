# ADR-021 — The checks speak to the chain; the bits become a signed cloud

**Status:** Accepted (2026-09-29)
**Depends on:** ADR-017 (reality-engine rules), ADR-019 (U1/U2), Unibit-Web ADR-0001/0002
(the ledger and the chain).

## 1. Why

The Unibit site's QμBLAR page has to show QμBLAR's numbers and QμBLAR's bits, and only those.
Today they live in stdout and in `build\ising_out_*`, which anyone could retype or redraw. Each
number and each bit field goes into the Unibit chain with the commit and check that produced it.
The page then draws only what the chain holds. Its own rule already says so: *a lattice render
is not a measurement*.

## 2. Decision

- **`src/ledger_out.hpp`.** A check records entries as it runs: id `qublar/<check>/<key>`,
  class MEASURED, the rule's verdict, the scene inputs, and the source (the ADR and rule).
  It writes `out/ledger/<check>.json`. Doubles are printed with `%.17g`, so the chain gets the
  exact binary value.
- **Artifacts.** A bit field is exported as `out/ledger/<name>.u8`:
  - one byte per voxel of the grid: 255 outside the domain, `round(254·p)` inside;
  - a `<name>.json` beside it with nx, voxel, lo, the truth voxels and the classify thresholds.

  The entry names the file. The batch tool adds its SHA-256 and size. A site fetches the file,
  hashes it with WebCrypto, and **refuses to draw it** unless the hash matches the chain.
- **`tools/ledger_batch.py`** makes the batch:
  1. It refuses a dirty tree.
  2. It builds the checks from HEAD.
  3. It runs them **one at a time** (GPU headroom).
  4. It collects the fragments and hashes the artifacts.
  5. It writes `out/ledger/qublar.json` with the commit.
- **Registered failures.** A check that exits non-zero refuses the batch, unless its failure is
  listed in `ledger/registered_failures.txt` with the ADR and prereg SHA that predicted it.
  - A registered failure is **signed as a FAIL**, not hidden and not waived.
  - Today the list has one entry: `check_fusion`, muons-only U1, ADR-019 §4.

## 3. What goes in

- `check_ising` (the void body run and its control), `check_mine` (4 runs), `check_ahead`,
  `check_gravity` (3 σ) and `check_fusion` (3 sensors, R1–R4). Each includes:
  - the evidence budget (data and prior nats);
  - confident / correct / off-truth (U1) and U2 misses;
  - localisation where claimed;
  - control false positives.
- **The first cloud** is `check_ising`'s pyramid posterior (128³ grid, 306,328 bits).

## 4. Refusals

- No page draws a bit field whose SHA-256 differs from the chain's.
- No unregistered failure is signed.
- No batch from a dirty tree.
- No number on the QμBLAR page without a ledger id.

## 5. First result (2026-09-29)

- **The batch.** `python tools/ledger_batch.py` at `d96b893` built and ran the five checks one at
  a time and produced 117 entries. Every number reproduced ADR-019 §5 exactly: 11/11, 37
  undecided, 47 with 16 U2 misses, 3 off the ore, 20 on the ore with 17 U2 misses, 85
  undecided, and 4 correct.
- **One FAIL, registered.** `qublar/check_fusion/muons/off_truth` = 3 was signed as FAIL with its
  prereg reference. Nothing else failed.
- **The signature.** Unibit chain block 3 (`c45868c3…`) holds all of it, plus the pyramid
  cloud: 2,097,152 bytes, SHA-256 `f86a406a6510…`.
- **The page.** The Unibit QμBLAR page draws the cloud in its Bits tab after verifying the hash
  in the browser. One flipped byte in the served file was refused with both hashes shown.
