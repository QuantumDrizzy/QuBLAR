"""QuBLAR -- render_khufu: figures for ADR-011 (check_khufu outputs). SYNTHETIC data only.

usage: python tools/render_khufu.py experiments/khufu
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

D = sys.argv[1] if len(sys.argv) > 1 else "experiments/khufu"
FIG = os.path.join(D, "figures")
os.makedirs(FIG, exist_ok=True)
TAG = "SYNTHETIC -- QuBLAR ADR-011 simulation, not real data"

meta = {}
for line in open(os.path.join(D, "meta.txt")):
    p = line.split()
    if len(p) >= 2:
        meta[p[0]] = p[1:]
NB, NTH, NAZ = int(meta["nb"][0]), int(meta["n_th"][0]), int(meta["n_az"][0])
THMAX = float(meta["th_max"][0])
dets = list(csv.DictReader(open(os.path.join(D, "detectors.csv"))))
ND = len(dets)
names = [d["name"] for d in dets]
dpos = np.array([[float(d["x"]), float(d["y"]), float(d["z"])] for d in dets])
area = np.array([float(d["area_m2"]) for d in dets])
days = np.array([float(d["days"]) for d in dets])
dset = [d["set"] for d in dets]


def f32(name, shape=None):
    a = np.fromfile(os.path.join(D, name), dtype=np.float32)
    return a.reshape(shape) if shape else a


acc = np.fromfile(os.path.join(D, "acceptance.u8"), dtype=np.uint8).reshape(ND, NB).astype(bool)
mod = {s: f32(f"rate_model_{s}.f32", (ND, NB)).astype(np.float64) for s in ["K0", "K1", "K1i", "K2", "K12"]}
real = {s: f32(f"rate_real_s1_{s}.f32", (ND, NB)).astype(np.float64) for s in ["K0", "K1", "K2"]}
realKB = f32("rate_real_KB.f32", (ND, NB)).astype(np.float64)

ith, iaz = np.divmod(np.arange(NB), NAZ)
th = (ith + 0.5) / NTH * THMAX
az = (iaz + 0.5) * 2 * np.pi / NAZ
bdir = np.stack([np.sin(th) * np.cos(az), np.sin(th) * np.sin(az), np.cos(th)], 1)

# ------------------------------------------------------------------ geometry helpers
HB, HO, HT = 115.15, 146.6, 138.5


def pyramid_edges():
    ht = HB * (1 - HT / HO)
    base = [(-HB, -HB, 0), (HB, -HB, 0), (HB, HB, 0), (-HB, HB, 0)]
    top = [(-ht, -ht, HT), (ht, -ht, HT), (ht, ht, HT), (-ht, ht, HT)]
    E = []
    for i in range(4):
        E.append((base[i], base[(i + 1) % 4]))
        E.append((top[i], top[(i + 1) % 4]))
        E.append((base[i], top[i]))
    return E


pts = f32("structures_points.f32").reshape(-1, 4)
LAB = {1: ("known chambers/corridors", "0.35", 1.0), 2: ("Big Void, horizontal (K1)", "red", 3.0),
       3: ("Big Void, inclined (K1i)", "orange", 1.5), 4: ("North Face Corridor (K2)", "royalblue", 4.0)}


def scene3d(ax, elev=18, azim=-60, show_legend=True, labels=True, pyramid=True):
    for a, b in (pyramid_edges() if pyramid else []):
        ax.plot(*zip(a, b), color="goldenrod", lw=0.8, alpha=0.7)
    for lab, (txt, col, sz) in LAB.items():
        p = pts[pts[:, 3] == lab]
        if lab == 1:
            p = p[::2]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=sz, c=col, alpha=0.35 if lab == 3 else 0.8, label=txt, depthshade=False,
                   edgecolors="none")
    for i, n in enumerate(names):
        col = "lime" if dset[i] == "BV" else "cyan"
        ax.scatter(*dpos[i], s=25, marker="^", c=col, edgecolors="k", linewidths=0.4, depthshade=False)
    if labels:
        for i in [0, 3, 4, 5, 6, 9, 12]:
            ax.text(*(dpos[i] + np.array([2, 0, 2])), names[i], fontsize=6)
    ax.set_xlim(-120, 120); ax.set_ylim(-120, 120); ax.set_zlim(0, 140)
    ax.set_box_aspect((240, 240, 140))
    ax.set_xlabel("x east [m]"); ax.set_ylabel("y north [m]"); ax.set_zlabel("z [m]")
    ax.view_init(elev=elev, azim=azim)
    if show_legend:
        h = [plt.Line2D([], [], ls="", marker="o", color=c, label=t) for t, c, _ in LAB.values()]
        h += [plt.Line2D([], [], ls="", marker="^", color="lime", mec="k", label="Big-Void detectors (QC + outside N face)"),
              plt.Line2D([], [], ls="", marker="^", color="cyan", mec="k", label="NFC detectors (descending corridor)")]
        ax.legend(handles=h, loc="upper left", fontsize=7)


fig = plt.figure(figsize=(15, 7.5))
ax = fig.add_subplot(1, 2, 1, projection="3d")
scene3d(ax)
ax.set_title("Khufu scene (full pyramid, cutaway = wireframe)")
ax = fig.add_subplot(1, 2, 2, projection="3d")
scene3d(ax, elev=12, azim=-25, show_legend=False, pyramid=False)
ax.set_xlim(-20, 30); ax.set_ylim(-15, 120); ax.set_zlim(-2, 75); ax.set_box_aspect((50, 135, 77))
ax.set_title("zoom: interior structures, Big Void hypotheses, NFC, detectors")
fig.suptitle(TAG, fontsize=9, color="gray")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "scene3d_cutaway.png"), dpi=140)
plt.close(fig)

# ------------------------------------------------------------------ excess maps
rng = np.random.default_rng(20260928)


def cell_map(dI, num, den, u, v, lim, cell):
    m = acc[dI]
    e = np.arange(-lim, lim + 1e-9, cell)
    S, _, _ = np.histogram2d(u[m], v[m], bins=[e, e], weights=num[m])
    B, _, _ = np.histogram2d(u[m], v[m], bins=[e, e], weights=den[m])
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = S / np.sqrt(B)
    Z[B <= 0] = np.nan
    return Z.T, e


def coords(dI):
    if dets[dI]["acceptance"] == "box_tan":
        return bdir[:, 0] / bdir[:, 2], bdir[:, 1] / bdir[:, 2], float(dets[dI]["a"]) * 1.02, 0.05, "tan theta_x (east)", "tan theta_y (north)"
    # cone: azimuth-from-south / elevation in degrees, relative to the axis
    aim = np.array([7.2, 18.0, 47.5]) - dpos[dI]
    aim /= np.linalg.norm(aim)
    ref_az = np.degrees(np.arctan2(aim[1], aim[0]))
    ref_el = np.degrees(np.arcsin(aim[2]))
    a = np.degrees(np.arctan2(bdir[:, 1], bdir[:, 0])) - ref_az
    a = (a + 180) % 360 - 180
    el = np.degrees(np.arcsin(bdir[:, 2])) - ref_el
    return a, el, 42.0, 2.0, "azimuth rel. to axis [deg]", "elevation rel. to axis [deg]"


for kind in ["expected", "seed1"]:
    fig, axs = plt.subplots(3, 5, figsize=(20, 12))
    axs = axs.ravel()
    for dI in range(ND):
        X = "K1" if dset[dI] == "BV" else "K2"
        w = area[dI] * days[dI]
        n0 = mod["K0"][dI] * w
        if kind == "expected":
            num = (mod[X][dI] - mod["K0"][dI]) * w
        else:
            lam = np.where(acc[dI], real[X][dI] * w, 0)
            num = rng.poisson(lam).astype(float) - n0
        u, v, lim, cell, xl, yl = coords(dI)
        Z, e = cell_map(dI, num, n0, u, v, lim, cell)
        vmax = np.nanpercentile(np.abs(Z), 99.5) if kind == "seed1" else np.nanmax(np.abs(Z))
        vmax = max(vmax, 3.0)
        im = axs[dI].imshow(Z, origin="lower", extent=[e[0], e[-1], e[0], e[-1]], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                            aspect="equal")
        plt.colorbar(im, ax=axs[dI], fraction=0.046, label="sigma per cell")
        axs[dI].set_title(f"{names[dI]} ({X} - K0, {int(days[dI])} d, {area[dI]} m2)", fontsize=9)
        axs[dI].set_xlabel(xl, fontsize=7); axs[dI].set_ylabel(yl, fontsize=7)
        axs[dI].tick_params(labelsize=7)
    for k in range(ND, 15):
        axs[k].axis("off")
    ttl = ("EXPECTED muon excess (void scene minus K0 model), sigma per cell = sum(N_X - N_K0)/sqrt(sum N_K0)"
           if kind == "expected" else
           "ONE REALISATION (seed 1 masonry + Poisson at the published exposure) vs K0 model: sum(d - N_K0)/sqrt(sum N_K0) per cell")
    axs[14].text(0, 0.5, ttl + "\n\nBig-Void set: K1 (horizontal hypothesis)\nNFC set: K2\nred = more muons than K0 (a void)",
                 fontsize=9, wrap=True, transform=axs[14].transAxes)
    fig.suptitle(TAG, fontsize=9, color="gray")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"excess_maps_{kind}.png"), dpi=110)
    plt.close(fig)

# ------------------------------------------------------------------ localisation scans
def grid_from(key):
    p = meta[key]
    lo = np.array([float(p[1]), float(p[2]), float(p[3])])
    st = np.array([float(p[5]), float(p[6]), float(p[7])])
    n = np.array([int(p[9]), int(p[10]), int(p[11])])
    return lo, st, n


loc = list(csv.DictReader(open(os.path.join(D, "localisation.csv"))))
truthKB = [float(x) for x in open(os.path.join(D, "blind_truth.txt")).readline().split()[1:4]]
estKB = [float(x) for x in open(os.path.join(D, "blind_estimate.txt")).readline().split()[1:4]]
scans = [("scan_dF_K1_s1.f32", "scanBV", "K1 seed 1 (Big Void, horizontal)", (7.2, 15.0, 60.0)),
         ("scan_dF_KB.f32", "scanBV", "KB BLIND (truth revealed after estimate)", tuple(truthKB)),
         ("scan_dF_K2_s1.f32", "scanNFC", "K2 seed 1 (NFC)", (7.2, 92.5, 21.0))]
fig, axs = plt.subplots(3, 3, figsize=(17, 14))
for r, (fn, key, ttl, tru) in enumerate(scans):
    lo, st, n = grid_from(key)
    F = f32(fn).reshape(n[2], n[1], n[0]).astype(np.float64)
    F = F - F.min()
    iz, iy, ix = np.unravel_index(np.argmin(F), F.shape)
    best = lo + st * np.array([ix, iy, iz])
    xs, ys, zs = [lo[i] + st[i] * np.arange(n[i]) for i in range(3)]
    Z = 2 * F  # -2 ln L ratio
    lv = np.log10(1 + Z)
    for c, (img, ext, xl, yl, tx, ty, bx, by) in enumerate([
        (lv[iz], [xs[0], xs[-1], ys[0], ys[-1]], "x [m]", "y [m]", tru[0], tru[1], best[0], best[1]),
        (lv[:, :, ix], [ys[0], ys[-1], zs[0], zs[-1]], "y [m]", "z [m]", tru[1], tru[2], best[1], best[2]),
        (lv[:, iy, :], [xs[0], xs[-1], zs[0], zs[-1]], "x [m]", "z [m]", tru[0], tru[2], best[0], best[2])]):
        im = axs[r, c].imshow(img, origin="lower", extent=ext, aspect="auto", cmap="viridis_r")
        plt.colorbar(im, ax=axs[r, c], label="log10(1 + 2 dF)")
        axs[r, c].plot(tx, ty, "r+", ms=16, mew=2, label="truth")
        axs[r, c].plot(bx, by, "wx", ms=10, mew=2, label="estimate")
        axs[r, c].set_xlabel(xl); axs[r, c].set_ylabel(yl)
        axs[r, c].legend(fontsize=7, loc="upper right")
    err = np.linalg.norm(best - np.array(tru))
    axs[r, 0].set_title(f"{ttl}\nestimate ({best[0]:.2f}, {best[1]:.2f}, {best[2]:.2f}), error {err:.2f} m", fontsize=9)
fig.suptitle("Template-scan localisation (data + K0 model only), slices through the minimum.  " + TAG, fontsize=9, color="gray")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "localisation_scans.png"), dpi=110)
plt.close(fig)

# ------------------------------------------------------------------ MLEM recon slices
if os.path.exists(os.path.join(D, "recon1m_K12_s1.f32")):
    sh = (142, 240, 240)
    rec = f32("recon1m_K12_s1.f32", sh)
    tru = f32("truth1m_K12.f32", sh)
    dom = np.fromfile(os.path.join(D, "domain1m.u8"), dtype=np.uint8).reshape(sh).astype(bool)
    ix = int((7.2 + 120) // 1)
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    for r, (vol, lab) in enumerate([(tru, "truth (K12, mean densities)"), (rec, "MLEM recon (K12 seed 1, f = 1, 20 it.)")]):
        v = np.where(dom, vol, np.nan)
        # y-z at x = 7.2 (passage plane): Big-Void box and NFC box
        for c, (ys, zs, ttl) in enumerate([((-30, 50), (40, 80), "Big-Void box, y-z at x = 7.5 m"),
                                           ((80, 104), (16, 28), "NFC box, y-z at x = 7.5 m")]):
            j0, j1 = ys[0] + 120, ys[1] + 120
            k0, k1 = zs[0] + 2, zs[1] + 2
            im = axs[r, c].imshow(v[k0:k1, j0:j1, ix], origin="lower", extent=[ys[0], ys[1], zs[0], zs[1]], cmap="magma",
                                  vmin=0, vmax=2.9, aspect="equal")
            plt.colorbar(im, ax=axs[r, c], fraction=0.03, label="g/cm3")
            axs[r, c].set_title(f"{lab}\n{ttl}", fontsize=9)
            axs[r, c].set_xlabel("y north [m]"); axs[r, c].set_ylabel("z [m]")
        k = 60 + 2
        im = axs[r, 2].imshow(v[k, 90:170, 100:140], origin="lower", extent=[-20, 20, -30, 50], cmap="magma", vmin=0, vmax=2.9,
                              aspect="equal")
        plt.colorbar(im, ax=axs[r, 2], fraction=0.03, label="g/cm3")
        axs[r, 2].set_title(f"{lab}\nx-y at z = 60.5 m", fontsize=9)
        axs[r, 2].set_xlabel("x east [m]"); axs[r, 2].set_ylabel("y north [m]")
    fig.suptitle("Density reconstruction inside the search boxes (figure only, not used by any rule).  " + TAG, fontsize=9,
                 color="gray")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "recon_slices_K12.png"), dpi=110)
    plt.close(fig)

# ------------------------------------------------------------------ significance vs exposure
rows = list(csv.DictReader(open(os.path.join(D, "summary_detection.csv"))))
fig, ax = plt.subplots(figsize=(8.5, 5.5))
cols = {"K0_vs_K1": "red", "K0_vs_K1i": "orange", "K0_vs_K2": "royalblue"}
for p, col in cols.items():
    rr = [r for r in rows if r["pair"] == p]
    f = [float(r["f"]) for r in rr]
    ax.plot(f, [float(r["Z_emp"]) for r in rr], "o-", color=col, label=f"{p}: Z_emp (8 seeds, masonry + scale nuisance)")
    ax.plot(f, [float(r["Z_asimov_nuis"]) for r in rr], "--", color=col, alpha=0.6, label=f"{p}: Asimov, scale nuisance only")
ax.axhline(5, color="k", lw=1, ls=":")
ax.axvline(1, color="gray", lw=1, ls=":")
ax.text(1.03, 6, "published exposure (f = 1)", fontsize=8, color="gray")
ax.text(0.105, 5.6, "Z = 5 (rule a)", fontsize=8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("exposure factor f (x each detector's published days)")
ax.set_ylabel("significance Z")
ax.set_title("Detection significance vs exposure (AUC = 1.000 at every point)")
ax.legend(fontsize=7, loc="center left")
fig.text(0.01, 0.01, TAG + ".  Z_emp is capped by the seed-to-seed masonry variance, which grows like f, so it does not rise with exposure.",
         fontsize=6.5, color="gray")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(FIG, "significance_vs_exposure.png"), dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ sim vs paper
pc = [r for r in csv.DictReader(open(os.path.join(D, "paper_comparison.csv"))) if r["scene"] in ("K1", "K2")]
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
labs, zs, zp, lower = [], [], [], []
for r in pc:
    if r["instrument"] == "G2":
        continue
    labs.append("G1+G2" if r["instrument"] == "G1" else r["instrument"])
    zs.append(float(r["Z_reg_sim"]))
    zp.append(float(r["Z_paper"].lstrip(">")))
    lower.append(r["Z_paper"].startswith(">"))
x = np.arange(len(labs))
axs[0].bar(x - 0.2, zs, 0.4, color="tab:blue", label="simulated Z_reg = S/sqrt(B) (anomaly region, f = 1)")
axs[0].bar(x + 0.2, zp, 0.4, color=["none" if l else "tab:gray" for l in lower], edgecolor="tab:gray", hatch=None,
           label="published (hollow = lower bound '> 10 sigma')")
for i, l in enumerate(lower):
    if l:
        axs[0].annotate("", xy=(x[i] + 0.2, zp[i] * 2.2), xytext=(x[i] + 0.2, zp[i]), arrowprops=dict(arrowstyle="->", color="gray"))
axs[0].axhline(5, color="k", ls=":", lw=1)
axs[0].set_yscale("log"); axs[0].set_xticks(x); axs[0].set_xticklabels(labs, rotation=45)
axs[0].set_ylabel("significance"); axs[0].legend(fontsize=8)
axs[0].set_title("Significance: simulation vs paper (Big Void: NE1..G1+G2; NFC: EM1..Degennes)")
labs2 = [r["instrument"] for r in pc]
ratio = [float(r["tracks_ratio"]) for r in pc]
x2 = np.arange(len(labs2))
axs[1].bar(x2, ratio, color=["tab:green" if 0.5 <= q <= 2 else "tab:red" for q in ratio])
axs[1].axhspan(0.5, 2.0, color="green", alpha=0.08, label="rule (d): within x2")
axs[1].axhline(1, color="k", lw=0.8)
axs[1].set_yscale("log"); axs[1].set_xticks(x2); axs[1].set_xticklabels(labs2, rotation=45)
axs[1].set_ylabel("simulated / published track count"); axs[1].legend(fontsize=8)
axs[1].set_title("Track counts in the acceptance (K0 model, published area x days)")
for i, q in enumerate(ratio):
    axs[1].text(i, q * 1.08, f"{q:.2f}", ha="center", fontsize=7)
fig.suptitle(TAG + ". Paper numbers: Morishima 2017 (arXiv:1711.01576 manuscript), Procureur 2023.", fontsize=9, color="gray")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "sim_vs_paper.png"), dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ orbit movie
fig = plt.figure(figsize=(8, 6.5))
ax = fig.add_subplot(111, projection="3d")
NFR = 90


def frame(i):
    ax.cla()
    scene3d(ax, elev=15 + 10 * np.sin(2 * np.pi * i / NFR), azim=-60 + 360 * i / NFR, show_legend=(i < 1000), labels=False)
    ax.set_title("QuBLAR Khufu scene (SYNTHETIC)", fontsize=10)
    return []


ani = animation.FuncAnimation(fig, frame, frames=NFR, blit=False)
try:
    ani.save(os.path.join(FIG, "orbit_khufu.mp4"), writer=animation.FFMpegWriter(fps=15, bitrate=2400), dpi=100)
except Exception as ex:  # noqa: BLE001
    print("mp4 failed:", ex)
ani.save(os.path.join(FIG, "orbit_khufu.gif"), writer=animation.PillowWriter(fps=12), dpi=60)
plt.close(fig)
print("figures written to", FIG)
for f in sorted(os.listdir(FIG)):
    print("  ", f)
