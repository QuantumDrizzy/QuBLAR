"""Render docs/benchmarks/*.png and vram_planning.csv from the committed experiment CSVs.

CPU only (matplotlib, Agg backend). It reads files and draws; it runs nothing on the GPU.

    python tools/render_benchmarks.py

Inputs (all committed):
    experiments/p1_bench/summary.csv      P1 (ADR-015) per-cell results, CPU only
    experiments/p2_gpu/summary.csv        P2 (ADR-016) per-device results, G1 and G22 only (run interrupted)
    experiments/p2_gpu/throughput.csv     P2 speedups
    docs/benchmarks/gpu_specs.csv         public card specs (verify before purchase)

Outputs (docs/benchmarks/):
    throughput_cpu_vs_gpu.png   p1_gap_vs_budget.png   p1_tts99.png   p2_tts99_cpu_vs_gpu.png
    vram_dense_n.png   vram_sparse_edges.png   gpu_speed_ceiling.png
    vram_planning.csv           CALCULATED ESTIMATES, NOT MEASURED
"""
import csv
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "benchmarks")

# Planning model constants (README section "VRAM planning"). Change them here only.
USABLE_FRACTION = 0.9            # fraction of VRAM usable for the problem
GIB = 2 ** 30                    # "16 GB" is treated as 16 GiB
BYTES_PER_UNDIRECTED_EDGE = 10   # CSR both directions: 2 x (int32 col + int8 weight)
INT32_EDGE_CAP = 2 ** 31 / 2     # 2^31 directed entries = 1.07e9 undirected edges
MEASURED_RATE = 1.11e10          # spin updates/s, RTX 5060 Ti, ADR-016 G1 (1.114e10)
MEASURED_SM = 36

ESTIMATE_STAMP = "CALCULATED ESTIMATE, NOT MEASURED"


def fnum(s):
    s = s.strip()
    if s.lower() in ("inf", "+inf"):
        return math.inf
    return float(s)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def read_throughput(path):
    """throughput.csv has an unquoted 'plan' field that contains commas; parse from both ends."""
    with open(path, encoding="utf-8") as f:
        rows = [ln.rstrip("\n").split(",") for ln in f if ln.strip()]
    tail = ["rate_cpu_per_s", "rate_gpu_per_s", "speedup", "wall_gpu_s", "wall_cpu_s",
            "kernel_ms", "launches", "sweeps_per_launch", "runs_gpu", "runs_cpu"]
    out = []
    for r in rows[1:]:
        d = {"instance": r[0], "plan": ",".join(r[1:-len(tail)]).strip()}
        d.update({k: fnum(v) for k, v in zip(tail, r[-len(tail):])})
        out.append(d)
    return out


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def stamp(ax, text=ESTIMATE_STAMP):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=15,
            color="red", alpha=0.18, rotation=20, weight="bold")


def err_from_ci(val, lo, hi, cap):
    """Asymmetric error bars; an infinite upper bound is drawn up to `cap` and flagged."""
    lo_e = val - lo
    hi_e = (min(hi, cap) - val) if math.isfinite(hi) else (cap - val)
    return lo_e, max(hi_e, 0.0), not math.isfinite(hi)


# ---------------------------------------------------------------- a. throughput
def chart_throughput(p2):
    inst = ["G1", "G22"]
    cpu = [next(r for r in p2 if r["instance"] == i and r["device"] == "cpu") for i in inst]
    gpu = [next(r for r in p2 if r["instance"] == i and r["device"] == "gpu") for i in inst]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = range(len(inst))
    w = 0.38
    bc = ax.bar([i - w / 2 for i in x], [fnum(r["rate_per_s"]) for r in cpu], w,
                label="CPU, 16 threads (P1 engine)", color="#5b7fa6")
    bg = ax.bar([i + w / 2 for i in x], [fnum(r["rate_per_s"]) for r in gpu], w,
                label="GPU, RTX 5060 Ti, 16384 replicas", color="#6aa84f")
    for bars in (bc, bg):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.15, f"{b.get_height():.3g}",
                    ha="center", fontsize=9)
    for i, (c, g) in enumerate(zip(cpu, gpu)):
        sp = fnum(g["rate_per_s"]) / fnum(c["rate_per_s"])
        ax.text(i, 3e10, f"{sp:.2f}x", ha="center", fontsize=10, weight="bold")
    ax.set_yscale("log")
    ax.set_ylim(1e8, 1e11)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{i}\n(n={c['n']}, m={c['m']})" for i, c in zip(inst, cpu)])
    ax.set_ylabel("spin updates / s (log)")
    ax.set_title("P2 (ADR-016, INTERRUPTED): same algorithm, CPU vs GPU\n"
                 "4000+200 sweeps, matched wall; measured 2026-09-28", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.grid(axis="y", which="both", alpha=0.25)
    save(fig, "throughput_cpu_vs_gpu.png")


# ---------------------------------------------------------------- b. P1 gap vs budget
def chart_p1_gap(p1):
    budgets = ["B1", "B10", "B100"]
    sweeps = {"B1": 420, "B10": 4200, "B100": 42000}
    inst = []
    for r in p1:
        if r["instance"] not in inst:
            inst.append(r["instance"])
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    cmap = plt.get_cmap("tab20")
    for k, name in enumerate(inst):
        rows = {r["budget"]: r for r in p1 if r["instance"] == name}
        ys = [fnum(rows[b]["gap_pct"]) for b in budgets]
        ax.plot([sweeps[b] for b in budgets], ys, marker="o", color=cmap(k % 20), label=name)
    ax.set_xscale("log")
    ax.set_xticks([sweeps[b] for b in budgets])
    ax.set_xticklabels([f"{b}\n({sweeps[b]} sweeps)" for b in budgets])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("best cut vs best-known (%)")
    ax.set_title("P1 (ADR-015): gap to best-known vs budget, CPU host SA, frozen t_hot=5\n"
                 "best of 64 runs (B1, B10) / 16 runs (B100); 0 = best-known reached (G1-G5 overlap near 0)", fontsize=10)
    ax.legend(ncol=2, fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    save(fig, "p1_gap_vs_budget.png")


# ---------------------------------------------------------------- c. P1 TTS99
def chart_p1_tts(p1):
    rows = [r for r in p1 if fnum(r["p_success"]) > 0]
    labels = [f"{r['instance']} {r['budget']}\np={fnum(r['p_success']):.3f}" for r in rows]
    vals = [fnum(r["tts99_s"]) for r in rows]
    cap = 300.0
    lo_e, hi_e, open_hi = [], [], []
    for r, v in zip(rows, vals):
        a, b, o = err_from_ci(v, fnum(r["tts99_ci_lo_s"]), fnum(r["tts99_ci_hi_s"]), cap)
        lo_e.append(a); hi_e.append(b); open_hi.append(o)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = list(range(len(rows)))
    ax.errorbar(x, vals, yerr=[lo_e, hi_e], fmt="o", color="#5b7fa6", capsize=3)
    for i, o in enumerate(open_hi):
        if o:
            ax.annotate("", xy=(i, cap * 1.6), xytext=(i, cap), arrowprops=dict(arrowstyle="->", color="#c0392b"))
    ax.set_yscale("log")
    ax.set_ylim(0.3, cap * 2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("TTS99 (s, log)")
    ax.set_title("P1 (ADR-015): TTS99 with bootstrap 95% CI, cells with p > 0 only\n"
                 "red arrow = CI upper bound is inf; t measured with 16 runs sharing the CPU. "
                 "All other cells: p = 0 (TTS99 = inf)", fontsize=9)
    ax.grid(axis="y", which="both", alpha=0.25)
    save(fig, "p1_tts99.png")


# ---------------------------------------------------------------- d. P2 TTS99
def chart_p2_tts(p2):
    inst = ["G1", "G22"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    cap = 100.0
    for k, (dev, col) in enumerate((("cpu", "#5b7fa6"), ("gpu", "#6aa84f"))):
        xs, ys, lo, hi = [], [], [], []
        for i, name in enumerate(inst):
            r = next(r for r in p2 if r["instance"] == name and r["device"] == dev)
            v = fnum(r["tts99_s"])
            a, b, o = err_from_ci(v, fnum(r["tts99_ci_lo_s"]), fnum(r["tts99_ci_hi_s"]), cap)
            xs.append(i + (k - 0.5) * 0.25); ys.append(v); lo.append(a); hi.append(b)
            ax.text(xs[-1] + 0.04, v, f" {v:.3g} s\n p={fnum(r['p_success']):.4f}", fontsize=7, va="center")
            if o:
                ax.annotate("", xy=(xs[-1], cap * 1.6), xytext=(xs[-1], cap),
                            arrowprops=dict(arrowstyle="->", color="#c0392b"))
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=col, capsize=4,
                    label={"cpu": "CPU, 16 threads", "gpu": "GPU, 16384 replicas"}[dev])
    ax.set_yscale("log")
    ax.set_ylim(5e-4, cap * 2)
    ax.set_xticks(range(len(inst)))
    ax.set_xticklabels(inst)
    ax.set_xlim(-0.6, 1.9)
    ax.set_ylabel("amortised TTS99 (s, log)")
    ax.set_title("P2 (ADR-016, INTERRUPTED, no verdict): amortised TTS99, 95% CI\n"
                 "t = batch wall / runs; NOT a single-answer latency (see README)", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", which="both", alpha=0.25)
    save(fig, "p2_tts99_cpu_vs_gpu.png")


# ---------------------------------------------------------------- e. VRAM planning
def planning(specs):
    rows = []
    for s in specs:
        vram = float(s["vram_gb"])
        cards = int(s["cards"])
        sm = int(s["sm_per_card"]) * cards
        watts = float(s["tgp_w_per_card"]) * cards
        usable = USABLE_FRACTION * vram * GIB
        n_dense = math.isqrt(int(usable))
        edges_mem = usable / BYTES_PER_UNDIRECTED_EDGE
        edges = min(edges_mem, INT32_EDGE_CAP)
        speed = MEASURED_RATE * sm / MEASURED_SM
        rows.append({
            "vram_gb": int(vram), "card": s["card"], "short": s["short"], "cards": cards, "total_sm": sm, "total_w": int(watts),
            "usable_bytes": f"{usable:.4g}", "dense_int8_n_max": n_dense,
            "sparse_edges_mem_limit": f"{edges_mem:.4g}", "sparse_edges_max": f"{edges:.4g}",
            "edges_capped_by_int32": "yes" if edges_mem > INT32_EDGE_CAP else "no",
            "speed_ceiling_per_s": f"{speed:.3g}", "speed_per_watt": f"{speed / watts:.3g}",
            "multi_gpu_code_needed": s["multi_gpu_code_needed"],
            "status": ESTIMATE_STAMP,
        })
    path = os.path.join(OUT, "vram_planning.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("# " + ESTIMATE_STAMP + ". Model and formulas: docs/benchmarks/README.md. "
                "Only the 16 GB card's 1.11e10/s is measured.\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", os.path.relpath(path, ROOT))
    return rows


def label(r):
    return f"{r['vram_gb']} GB\n{r['short']}"


def chart_vram(rows):
    v = list(range(len(rows)))  # categorical positions, one per VRAM size
    # dense
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    n = [r["dense_int8_n_max"] for r in rows]
    ax.plot(v, n, marker="o", color="#8e44ad")
    for x, y, r in zip(v, n, rows):
        ax.text(x, y * 1.03, f"{y/1e3:.0f}k", ha="center", fontsize=8)
    ax.axhline(2000, color="grey", ls=":", lw=1)
    ax.text(v[0], 2000 * 1.1, "K2000 (n = 2000)", fontsize=7, color="grey")
    ax.set_xticks(v)
    ax.set_xticklabels([label(r) for r in rows], fontsize=8)
    ax.set_ylabel("max N, dense int8 J (full N x N matrix)")
    ax.set_yscale("log")
    ax.set_title("Max dense problem size vs VRAM: N_max = sqrt(0.9 * VRAM)\n" + ESTIMATE_STAMP, fontsize=9)
    ax.grid(alpha=0.25, which="both")
    stamp(ax)
    save(fig, "vram_dense_n.png")
    # sparse
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    em = [float(r["sparse_edges_mem_limit"]) for r in rows]
    ec = [float(r["sparse_edges_max"]) for r in rows]
    ax.plot(v, em, marker="o", ls="--", color="#95a5a6", label="memory limit: 0.9 * VRAM / 10 B per edge")
    ax.plot(v, ec, marker="o", color="#d35400", label="usable today (min of memory, int32 cap)")
    ax.axhline(INT32_EDGE_CAP, color="#c0392b", lw=1.2)
    ax.text(v[-1], INT32_EDGE_CAP * 1.08, "int32 cap 1.07e9 edges (until 64-bit indices, P4)",
            ha="right", fontsize=8, color="#c0392b")
    ax.axhline(40000, color="grey", ls=":", lw=1)
    ax.text(v[0], 40000 * 1.3, "G81 (m = 40000)", fontsize=7, color="grey")
    ax.set_yscale("log")
    ax.set_xticks(v)
    ax.set_xticklabels([label(r) for r in rows], fontsize=8)
    ax.set_ylabel("max undirected edges, sparse CSR")
    ax.set_title("Max sparse problem size vs VRAM (int32 col + int8 weight, both directions)\n"
                 + ESTIMATE_STAMP, fontsize=9)
    ax.legend(fontsize=7, loc="center right")
    ax.grid(alpha=0.25, which="both")
    stamp(ax)
    save(fig, "vram_sparse_edges.png")
    # speed
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    sp = [float(r["speed_ceiling_per_s"]) for r in rows]
    spw = [float(r["speed_per_watt"]) for r in rows]
    cols = ["#6aa84f" if r["vram_gb"] == 16 else ("#f39c12" if r["multi_gpu_code_needed"] == "yes" else "#7f8c8d")
            for r in rows]
    lab = [f"{r['vram_gb']} GB\n{r['short']}\n{r['total_sm']} SM\n{r['total_w']} W" for r in rows]
    a1.bar(range(len(rows)), sp, color=cols)
    for i, y in enumerate(sp):
        a1.text(i, y * 1.02, f"{y:.3g}", ha="center", fontsize=8)
    a1.set_ylabel("spin updates / s (ceiling)")
    a1.set_title("Speed ceiling = 1.11e10 * SM / 36 (optimistic linear)", fontsize=9)
    a2.bar(range(len(rows)), spw, color=cols)
    for i, y in enumerate(spw):
        a2.text(i, y * 1.02, f"{y:.3g}", ha="center", fontsize=8)
    a2.set_ylabel("spin updates / s per board watt (ceiling)")
    a2.set_title("Speed per watt (ceiling / TGP)", fontsize=9)
    for a in (a1, a2):
        a.set_xticks(range(len(rows)))
        a.set_xticklabels(lab, fontsize=7)
        a.grid(axis="y", alpha=0.25)
        stamp(a)
    fig.suptitle("Green = measured card (16 GB, 1.11e10/s on G1). Grey/orange = extrapolated, "
                 "NOT MEASURED; orange needs multi-GPU code that does not exist", fontsize=9, y=1.04)
    save(fig, "gpu_speed_ceiling.png")


def markdown_table(rows):
    print("\n| VRAM | example card | SM | W | dense int8 N_max | sparse edges max | int32-capped | "
          "speed ceiling (/s) | per W | multi-GPU code |")
    print("|---:|---|---:|---:|---:|---:|---|---:|---:|---|")
    for r in rows:
        print(f"| {r['vram_gb']} GB | {r['card']} | {r['total_sm']} | {r['total_w']} | "
              f"{r['dense_int8_n_max']:,} | {float(r['sparse_edges_max']):.3g} | {r['edges_capped_by_int32']} | "
              f"{float(r['speed_ceiling_per_s']):.3g} | {float(r['speed_per_watt']):.3g} | "
              f"{r['multi_gpu_code_needed']} |")


def main():
    os.makedirs(OUT, exist_ok=True)
    p1 = read_csv(os.path.join(ROOT, "experiments", "p1_bench", "summary.csv"))
    p2 = read_csv(os.path.join(ROOT, "experiments", "p2_gpu", "summary.csv"))
    thr = read_throughput(os.path.join(ROOT, "experiments", "p2_gpu", "throughput.csv"))
    specs = read_csv(os.path.join(OUT, "gpu_specs.csv"))
    # Cross-check: summary.csv rates and throughput.csv speedups must agree.
    for t in thr:
        c = next(r for r in p2 if r["instance"] == t["instance"] and r["device"] == "cpu")
        g = next(r for r in p2 if r["instance"] == t["instance"] and r["device"] == "gpu")
        sp = fnum(g["rate_per_s"]) / fnum(c["rate_per_s"])
        if abs(sp - t["speedup"]) > 1e-6 * sp:
            sys.exit(f"speedup mismatch for {t['instance']}: {sp} vs {t['speedup']}")
        print(f"{t['instance']}: speedup {t['speedup']:.2f}x (summary and throughput agree)")
    chart_throughput(p2)
    chart_p1_gap(p1)
    chart_p1_tts(p1)
    chart_p2_tts(p2)
    rows = planning(specs)
    chart_vram(rows)
    markdown_table(rows)


if __name__ == "__main__":
    main()
