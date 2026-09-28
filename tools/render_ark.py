"""Render the ADR-010 (check_ark) outputs: scenes, detector layout, transmission and
difference maps, MLEM density slices, significance vs exposure, and an orbit movie.

Usage: python tools/render_ark.py experiments/ark
Reads only files written by build/check_ark.exe; computes no new statistics except
per-bin display quantities (transmission = rate / open flux, per-bin Poisson pulls).
"""
import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

OUT = sys.argv[1] if len(sys.argv) > 1 else "experiments/ark"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

RATE = 1.44e7            # muons m^-2 day^-1 (as in check_ark)
NTH, NAZ, THMAX = 70, 360, 70.0
DET = np.array([[-45, -15, -20], [0, -15, -20], [45, -15, -20],
                [-45, 15, -20], [0, 15, -20], [45, 15, -20]], float)
HYPS = ["A", "B1", "B2", "B3", "B3x"]
TITLE = {"A": "H_A natural (layered syncline)", "B1": "H_B1 hull, air compartments",
         "B2": "H_B2 hull, sediment-filled", "B3": "H_B3 petrified hull, filled",
         "B3x": "B3x petrified walls only (POST-HOC)"}
LO = np.array([-104.0, -72.0, -22.0])
N1 = (28, 144, 208)      # z, y, x at 1 m


def f32(name, shape=None):
    a = np.fromfile(os.path.join(OUT, name), dtype=np.float32)
    return a.reshape(shape) if shape else a


def mound_halfwidth(x):
    u = x / 80.0
    w = np.where(x < 0, 22.5 * np.sqrt(np.clip(1 - u * u, 0, None)), 22.5 * (1 - u * u))
    return np.where(np.abs(u) < 1, w, 0.0)


def surface_z(x, y):
    w = mound_halfwidth(x)
    r = np.where(w > 0, y / np.maximum(w, 1e-6), 2.0)
    return np.where(np.abs(r) < 1, 4.0 * np.sqrt(np.clip(1 - r * r, 0, None)), 0.0)


def hull_halfbeam(x):
    return 13.0 * np.clip((78.5 - np.abs(x)) / 15.0, None, 1.0)


open_frac = f32("open_fraction.f32").reshape(NTH, NAZ)
truth = {h: f32(f"truth1m_{h}.f32", N1) for h in HYPS}
recon = {h: f32(f"recon1m_{h}.f32", N1) for h in HYPS}
domain = np.fromfile(os.path.join(OUT, "domain1m.u8"), dtype=np.uint8).reshape(N1)
xs = LO[0] + np.arange(N1[2]) + 0.5
ys = LO[1] + np.arange(N1[1]) + 0.5
zs = LO[2] + np.arange(N1[0]) + 0.5


# ------------------------------------------------------------------ 3D scenes
def draw_scene(ax, h, elev=24, azim=-60):
    ax.cla()
    # mound surface, front half (y < 0) removed = cutaway
    gx = np.linspace(-80, 80, 161)
    gy = np.linspace(0, 23, 24)
    X, Y = np.meshgrid(gx, gy)
    Z = surface_z(X, Y)
    Z = np.where(np.abs(Y) <= mound_halfwidth(X), Z, np.nan)
    ax.plot_surface(X, Y, Z, color="#b59a6b", alpha=0.25, linewidth=0, shade=True)
    # cut face y = 0.5: density section from the 1 m truth
    j = np.argmin(np.abs(ys - 0.5))
    sec = truth[h][:, j, :]
    XX, ZZ = np.meshgrid(xs, zs)
    keep = (np.abs(XX) < 95) & (ZZ > -21)
    cmap = plt.get_cmap("cividis")
    fc = cmap(np.clip(sec / 2.8, 0, 1))
    fc[~keep] = (1, 1, 1, 0)
    fc[sec < 0.05] = (0.85, 0.93, 1.0, 0.0)
    ax.plot_surface(XX, np.full_like(XX, 0.5), ZZ, facecolors=fc, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    # hidden structure (voxels that differ from the paired H_A realisation), back half
    if h != "A":
        d = truth[h] != truth["A"]
        kz, jy, ix = np.nonzero(d)
        sel = ys[jy] >= 0
        v = truth[h][kz, jy, ix][sel]
        px, py, pz = xs[ix][sel], ys[jy][sel], zs[kz][sel]
        wall = (v > 0.1) & ((v < 1.2) | (v > 2.3))
        fill = (v >= 1.2) & (v <= 2.3)
        ax.scatter(px[wall], py[wall], pz[wall], s=2, c="#6b3b1a" if h in ("B1", "B2") else "#3a3a3a",
                   alpha=0.55, depthshade=True, marker="s", linewidths=0)
        if fill.any():
            ax.scatter(px[fill][::3], py[fill][::3], pz[fill][::3], s=1.5, c="#d9b36c", alpha=0.12,
                       marker="s", linewidths=0)
    ax.scatter(DET[:, 0], DET[:, 1], DET[:, 2], s=40, c="red", marker="s", depthshade=False)
    ax.set_xlim(-100, 100); ax.set_ylim(-60, 60); ax.set_zlim(-22, 8)
    ax.set_box_aspect((200, 120, 60))
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(TITLE[h] + "\ncutaway: section y=0, hidden structure y>0; red = detectors", fontsize=9)


for h in HYPS:
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111, projection="3d")
    draw_scene(ax, h)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"scene3d_{h}.png"), dpi=130)
    plt.close(fig)

# ------------------------------------------------------------------ layout
fig, axs = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [1.2, 1]})
ax = axs[0]
gx = np.linspace(-80, 80, 400)
w = mound_halfwidth(gx)
ax.fill_between(gx, -w, w, color="#b59a6b", alpha=0.4, label="mound planform (160 x 45 m, relief 4 m)")
hx = np.linspace(-78.5, 78.5, 400)
hb = hull_halfbeam(hx)
ax.plot(hx, hb, "k-", lw=1.2, label="hull outline (157 x 26 m), H_B*")
ax.plot(hx, -hb, "k-", lw=1.2)
ax.add_patch(plt.Rectangle((20 - 4, -4), 8, 8, fill=False, ec="b", lw=1, ls="--", label="void sweep centre (8 m shown)"))
for i, d in enumerate(DET):
    ax.plot(d[0], d[1], "rs", ms=9)
    ax.annotate(f"D{i}", (d[0] + 2, d[1] + 2), color="r")
    ax.add_patch(plt.Circle((d[0], d[1]), 24 * np.tan(np.radians(70)), fill=False, ec="r", lw=0.4, alpha=0.5))
ax.set_aspect("equal"); ax.set_xlim(-104, 104); ax.set_ylim(-72, 72)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("Top view: 6 x 1 m^2 detectors at z = -20 m (circles: 70 deg zenith footprint at z = +4 m)")
ax.legend(loc="lower left", fontsize=8)
ax = axs[1]
j = np.argmin(np.abs(ys - (-6.5)))
im = ax.imshow(truth["B1"][:, j, :], origin="lower", extent=[LO[0], LO[0] + 208, LO[2], LO[2] + 28],
               cmap="cividis", vmin=0, vmax=2.8, aspect="auto")
for d in DET[:3]:
    ax.plot(d[0], d[2], "rs", ms=8)
    for th in (-70, -35, 0, 35, 70):
        t = np.radians(th)
        ax.plot([d[0], d[0] + 26 * np.tan(t)], [d[2], 6], "r-", lw=0.5, alpha=0.5)
ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
ax.set_title("Side section y = -6.5 m through H_B1 (seed 1, 1 m display sampling) with ray fans of D0-D2 (+-70 deg; D0-D2 sit at y = -15)")
fig.colorbar(im, ax=ax, label="density g/cm^3")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "detector_layout.png"), dpi=130)
plt.close(fig)


# ------------------------------------------------------------------ maps
def rate(kind, h):
    return f32(f"rate_{kind}_{h}.f32" if kind == "model" else f"rate_real_s1_{h}.f32").reshape(6, NTH, NAZ)


def polar(ax, data, title, cmap, vmin, vmax, label):
    th = np.radians(np.linspace(0, 360, NAZ + 1))
    r = np.linspace(0, THMAX, NTH + 1)
    T, R = np.meshgrid(th, r)
    m = ax.pcolormesh(T, R, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
    ax.set_title(title, fontsize=9)
    ax.set_yticks([20, 40, 60]); ax.set_yticklabels(["20", "40", "60 deg"], fontsize=7)
    ax.tick_params(labelsize=7)
    plt.colorbar(m, ax=ax, fraction=0.046, pad=0.08, label=label)


with np.errstate(divide="ignore", invalid="ignore"):
    trans = {h: rate("real", h) / (open_frac[None] * RATE) for h in HYPS}
DSEL = 1
fig, axs = plt.subplots(1, 5, figsize=(22, 4.8), subplot_kw={"projection": "polar"})
for ax, h in zip(axs, HYPS):
    polar(ax, trans[h][DSEL], f"{h}: transmission, D{DSEL} (seed 1)", "magma", 0, 0.15, "T")
fig.suptitle("Muon transmission maps seen from detector D1 (x=0, y=-15, z=-20); azimuth 0 = +x (bow), radius = zenith", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "transmission_maps_D1.png"), dpi=120)
plt.close(fig)

fig, axs = plt.subplots(2, 3, figsize=(15, 9), subplot_kw={"projection": "polar"})
for d, ax in enumerate(axs.flat):
    polar(ax, trans["A"][d], f"H_A seed 1, D{d} ({DET[d,0]:.0f}, {DET[d,1]:.0f})", "magma", 0, 0.15, "T")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "transmission_maps_A_all_detectors.png"), dpi=110)
plt.close(fig)

E = 30.0
fig, axs = plt.subplots(2, 4, figsize=(20, 9), subplot_kw={"projection": "polar"})
for c, h in enumerate(["B1", "B2", "B3", "B3x"]):
    for r_, d in enumerate([1, 4]):
        nA = rate("real", "A")[d] * E
        nH = rate("real", h)[d] * E
        with np.errstate(divide="ignore", invalid="ignore"):
            pull = (nH - nA) / np.sqrt(nA)
        lim = {"B1": 60, "B2": 20, "B3": 10, "B3x": 4}[h]
        polar(axs[r_, c], pull, f"{h} - A, D{d}, 30 d: (N_X - N_A)/sqrt(N_A)", "RdBu_r", -lim, lim, "sigma per bin")
fig.suptitle("Difference maps (paired seed-1 backgrounds, expected counts at 30 days, 1 m^2 detector): red = more muons = less mass", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "difference_maps_30d.png"), dpi=110)
plt.close(fig)

# ------------------------------------------------------------------ recon slices
kz = np.argmin(np.abs(zs - (-5.5)))
jy = np.argmin(np.abs(ys - 0.5))
fig, axs = plt.subplots(len(HYPS), 4, figsize=(20, 3.1 * len(HYPS)))
for r_, h in enumerate(HYPS):
    for c, (vol, name) in enumerate([(truth[h], "truth"), (recon[h], "MLEM recon")]):
        hs = np.where(domain[kz] > 0, vol[kz], np.nan)
        axs[r_, c].imshow(hs, origin="lower", extent=[-104, 104, -72, 72], cmap="cividis", vmin=0, vmax=2.8)
        axs[r_, c].set_xlim(-85, 85); axs[r_, c].set_ylim(-26, 26)
        axs[r_, c].set_title(f"{h} {name}: z = {zs[kz]:.1f} m", fontsize=9)
        ls = np.where(domain[:, jy, :] > 0, vol[:, jy, :], np.nan)
        im = axs[r_, c + 2].imshow(ls, origin="lower", extent=[-104, 104, -22, 6], cmap="cividis",
                                   vmin=0, vmax=2.8, aspect="auto")
        axs[r_, c + 2].set_xlim(-85, 85); axs[r_, c + 2].set_ylim(-17, 5)
        axs[r_, c + 2].set_title(f"{h} {name}: y = {ys[jy]:.1f} m", fontsize=9)
fig.colorbar(im, ax=axs, fraction=0.015, label="density g/cm^3")
fig.suptitle("Density slices inside the formation body: truth (seed 1) vs MLEM from 30 days x 6 m^2 (30 iterations, 1 m voxels)", fontsize=11)
fig.savefig(os.path.join(FIG, "recon_slices_30d.png"), dpi=100, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ significance
rows = list(csv.DictReader(open(os.path.join(OUT, "summary.csv"))))
def series(pair, key):
    rr = [r for r in rows if r["pair"] == pair]
    return np.array([float(r["exposure_days"]) for r in rr]), np.array([float(r[key]) for r in rr])

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
ax = axs[0]
for p, col in [("A_vs_B1", "C0"), ("A_vs_B2", "C1"), ("A_vs_B3", "C2"), ("A_vs_B3x", "C7")]:
    x, z = series(p, "Z_emp")
    _, zi = series(p, "Z_asimov_nuis")
    ls = "--" if p.endswith("B3x") else "-"
    ax.plot(x, z, ls, marker="o", color=col, label=p.replace("_vs_", " vs ") + (" (post-hoc)" if p.endswith("B3x") else ""))
    ax.plot(x, zi, ":", color=col, lw=1)
ax.axhline(5, color="k", lw=1); ax.axvline(30, color="gray", lw=0.8, ls="--")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("exposure (days, 6 x 1 m^2)"); ax.set_ylabel("Z")
ax.set_title("Hull hypotheses: Z_emp over 8 seeds (solid)\nAsimov with density nuisance, known background (dotted)", fontsize=9)
ax.legend(fontsize=8)
ax = axs[1]
Ls = [1, 2, 3, 4, 6, 8]
for e, m in zip([1, 7, 30, 90], "os^D"):
    zz = [float([r for r in rows if r["pair"] == f"A_vs_V{L}m" and r["exposure_days"] == str(e)][0]["Z_emp"]) for L in Ls]
    ax.plot(Ls, zz, marker=m, label=f"{e} d")
ax.axhline(5, color="k", lw=1)
ax.set_xlabel("air void cube edge L (m), centred at (20, 0, -5)"); ax.set_ylabel("Z_emp (8 seeds)")
ax.set_title("Void sweep: empirical separation vs size", fontsize=9)
ax.legend(fontsize=8)
ax = axs[2]
q = list(csv.DictReader(open(os.path.join(OUT, "q_values.csv"))))
for i, p in enumerate(["A_vs_B3", "A_vs_V2m", "A_vs_V3m", "A_vs_B3x"]):
    qa = [float(r["q_under_A"]) for r in q if r["pair"] == p and r["exposure_days"] == "30"]
    qx = [float(r["q_under_X"]) for r in q if r["pair"] == p and r["exposure_days"] == "30"]
    s = max(np.std(qa), 1e-9)
    ax.scatter(np.array(qa) / s, np.full(len(qa), i) - 0.1, c="C0", s=18, label="data under H_A" if i == 0 else None)
    ax.scatter(np.array(qx) / s, np.full(len(qx), i) + 0.1, c="C3", s=18, label="data under H_X" if i == 0 else None)
ax.set_yticks(range(4)); ax.set_yticklabels(["A vs B3", "A vs V2m", "A vs V3m", "A vs B3x"])
ax.set_xlabel("q / sd(q under A), 30 d"); ax.set_title("Per-seed test statistic (8 seeds each)", fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "significance_vs_exposure.png"), dpi=120)
plt.close(fig)

# ------------------------------------------------------------------ orbit
fig = plt.figure(figsize=(8, 4.8))
ax = fig.add_subplot(111, projection="3d")
def frame(i):
    draw_scene(ax, "B1", elev=22, azim=-60 + i * 10)
    return []
anim = animation.FuncAnimation(fig, frame, frames=36, blit=False)
try:
    anim.save(os.path.join(FIG, "orbit_B1.mp4"), writer=animation.FFMpegWriter(fps=8, bitrate=1800), dpi=100)
except Exception as e:  # noqa
    print("mp4 failed:", e)
try:
    anim.save(os.path.join(FIG, "orbit_B1.gif"), writer=animation.PillowWriter(fps=8), dpi=60)
except Exception as e:  # noqa
    print("gif failed:", e)
plt.close(fig)
print("figures in", FIG)
for f in sorted(os.listdir(FIG)):
    print(" ", f, os.path.getsize(os.path.join(FIG, f)))
