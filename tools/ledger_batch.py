"""QuBLAR's numbers and bit fields as one batch for the Unibit chain (ADR-021).

    python tools/ledger_batch.py            # build, run the checks one at a time, write the batch
    python tools/ledger_batch.py --no-build # reuse build\\*.exe (only if built from HEAD)

Refusals:
- a dirty tree, before or after the runs;
- a check that fails to build;
- a check that exits non-zero, unless every FAIL entry it wrote is listed in
  ledger/registered_failures.txt (a failure predicted by a frozen prereg is
  signed as a FAIL, never hidden and never waived);
- an artifact that is missing.

Output: out/ledger/qublar.json (the batch) and out/ledger/artifacts/ (the files
the sites fetch; each one's SHA-256 is in its entry).
"""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "ledger"
# Light checks first; the GPU-heavy ones last, one at a time (GPU headroom rule).
CHECKS = ["check_ahead", "check_gravity", "check_fusion", "check_mine", "check_ising"]


def die(msg: str) -> None:
    print(f"ledger_batch: {msg}", file=sys.stderr)
    sys.exit(1)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def clean_tree() -> bool:
    return git("status", "--porcelain") == ""


def registered() -> dict[str, dict[str, str]]:
    reg: dict[str, dict[str, str]] = {}
    for line in (ROOT / "ledger" / "registered_failures.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        check, entry_id, prereg, what = [x.strip() for x in line.split("|")]
        reg.setdefault(check, {})[entry_id] = f"{prereg}: {what}"
    return reg


def main() -> None:
    if not clean_tree():
        die("the tree is dirty; a number from uncommitted code is not reproducible")
    commit = git("rev-parse", "--short=12", "HEAD")

    if "--no-build" not in sys.argv:
        for c in CHECKS:
            print(f"  build {c}", flush=True)
            r = subprocess.run(["cmd", "/c", str(ROOT / "tools" / "build_one.bat"), c], cwd=ROOT)
            if r.returncode != 0:
                die(f"{c} did not build")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    reg = registered()
    entries: list[dict] = []

    for c in CHECKS:
        print(f"  run   {c}", flush=True)
        r = subprocess.run([str(ROOT / "build" / f"{c}.exe")], cwd=ROOT, capture_output=True, text=True)
        tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        print(f"        exit {r.returncode}  {tail}", flush=True)
        frag_path = OUT / f"{c}.json"
        if not frag_path.exists():
            die(f"{c} wrote no ledger fragment")
        frag = json.loads(frag_path.read_text(encoding="utf-8"))
        fails = [e["id"] for e in frag["entries"] if e["verdict"] == "FAIL"]
        if r.returncode != 0:
            unregistered = [f for f in fails if f not in reg.get(c, {})]
            if not fails or unregistered:
                die(f"{c} exited {r.returncode} with unregistered failure(s): {unregistered or '(no FAIL entry)'}")
        for e in frag["entries"]:
            if e["verdict"] == "FAIL" and e["id"] in reg.get(c, {}):
                e["source"] += f" [REGISTERED FAIL: {reg[c][e['id']]}]"
            path = e.pop("artifact_path", None)
            if path:
                src = ROOT / path
                if not src.exists():
                    die(f"{e['id']}: artifact {path} is missing")
                data = src.read_bytes()
                published = f"qublar/{src.name}"
                dst = OUT / "artifacts" / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
                e["artifact"] = {"path": published, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            entries.append(e)

    if not clean_tree():
        die("a check changed tracked files; nothing written")
    batch = {"repo": "qublar", "commit": commit, "dirty": False, "producer": "tools/ledger_batch.py", "entries": entries}
    (OUT / "qublar.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    n_fail = sum(e["verdict"] == "FAIL" for e in entries)
    print(f"\nledger -> {OUT / 'qublar.json'}  ({len(entries)} entries, {n_fail} registered FAIL, commit {commit})")


if __name__ == "__main__":
    main()
