"""ADR-014 figures: 3D scene (+ orbit MP4), excess maps, detectability map, exposure curves,
template-scan slices, MLEM recon slices. Usage: python tools/render_ree_muo.py <experiment dir> [--no-mp4]"""
import csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

D = sys.argv[1]
MP4 = "--no-mp4" not in sys.argv
FIG = os.path.join(D, "figures")
os.makedirs(FIG, exist_ok=True)
meta = {}
for l in open(os.path.join(D, "meta.txt")):
    p = l.split()
    if p:
        meta[p[0]] = p[1:]
NTH, NAZ, THMAX = int(meta["n_th"][0]), int(meta["n_az"][0]), float(meta["th_max"][0])
NB = NTH * NAZ
EX = int(meta["example_cell"][0])
dets = list(csv.DictReader(open(os.path.join(D, "detectors.csv"))))
DP = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in dets])
cells = {int(r["cell"]): r for r in csv.DictReader(open(os.path.join(D, "cells.csv")))}
rules = json.load(open(os.path.join(D, "ree_muo_rules.json")))
bt = {l.split()[0]: l.split()[1:] for l in open(os.path.join(D, "blind_truth.txt")) if l.strip()}
be = {l.split()[0]: l.split()[1:] for l in open(os.path.join(D, "blind_estimate.txt")) if l.strip()}
BT = np.array([float(x) for x in bt["truth"]]); BE = np.array([float(x) for x in be["estimate"]])
ld = lambda n, dt=np.float32: np.fromfile(os.path.join(D, n), dtype=dt)

ith = np.arange(NB) // NAZ; iaz = np.arange(NB) % NAZ
th = (ith + 0.5) / NTH * THMAX; az = (iaz + 0.5) * 2 * np.pi / NAZ
tx = np.tan(th) * np.cos(az); ty = np.tan(th) * np.sin(az)

def cube_faces(c, L):
    h = L / 2; x0, x1 = c[0] - h, c[0] + h; y0, y1 = c[1] - h, c[1] + h; z0, z1 = c[2] - h, c[2] + h
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    f = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
    return [[v[i] for i in q] for q in f]

def draw_scene(ax, show_blind=True):
    # surface outline
    s = 200
    ax.plot([-s, s, s, -s, -s], [-s, -s, s, s, -s], [0] * 5, color="0.5", lw=0.8)
    ax.add_collection3d(Poly3DCollection([[[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]]], alpha=0.06, facecolor="tab:green"))
    ax.text(-s, -s, 5, "surface z = 0", fontsize=7)
    # drift
    xs = [-200, 200]
    for yy in (-2.5, 2.5):
        for zz in (-201, -196):
            ax.plot(xs, [yy, yy], [zz, zz], color="k", lw=0.8)
    ax.text(150, 0, -212, "drift (5 x 5 m)", fontsize=7)
    ax.scatter(DP[:, 0], DP[:, 1], DP[:, 2], color="tab:red", s=18, depthshade=False)
    # acceptance pyramid of D2 and D3
    for p in DP[1:3]:
        for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            ax.plot([p[0], p[0] + sx * 200], [p[1], p[1] + sy * 200], [p[2], 0], color="tab:red", lw=0.4, alpha=0.4)
    cols = {40: "tab:blue", 80: "tab:cyan", 120: "tab:orange", 160: "tab:purple"}
    for d in (40, 80, 120, 160):
        ax.add_collection3d(Poly3DCollection(cube_faces([0, 20, -d], 20), alpha=0.35, facecolor=cols[d], edgecolor="k", lw=0.3))
        ax.text(12, 20, -d, f"L=20 m, depth {d} m", fontsize=6)
    if show_blind:
        for q in cube_faces(BT, 20):
            q = np.array(q + [q[0]])
            ax.plot(q[:, 0], q[:, 1], q[:, 2], color="tab:red", lw=1.0)
        ax.scatter([BE[0]], [BE[1]], [BE[2]], marker="x", color="tab:red", s=30)
    ax.set_xlim(-200, 200); ax.set_ylim(-200, 200); ax.set_zlim(-210, 0)
    ax.set_xlabel("x east (m)"); ax.set_ylabel("y north (m)"); ax.set_zlabel("z (m)")
    ax.set_box_aspect((2, 2, 1.05))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
draw_scene(ax)
ax.view_init(elev=18, azim=-60)
ax.set_title("ADR-014 (synthetic): drift at 200 m with four 1 m$^2$ detectors (red), |tan|<=1 acceptance (D2, D3 shown)\n"
             "swept dense cubes at depth 40-160 m (L = 20 m shown); blind body (red wire, truth revealed after) and estimate (x)", fontsize=9)
fig.savefig(os.path.join(FIG, "fig_muo_scene_3d.png"), dpi=140, bbox_inches="tight"); plt.close(fig)

if MP4:
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    fig = plt.figure(figsize=(8, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    draw_scene(ax)
    ax.set_title("ADR-014 synthetic scene (orbit)", fontsize=9)
    def upd(i):
        ax.view_init(elev=15 + 10 * np.sin(2 * np.pi * i / 120), azim=-60 + 3 * i)
        return []
    anim = FuncAnimation(fig, upd, frames=120, interval=50)
    anim.save(os.path.join(FIG, "fig_muo_orbit.mp4"), writer=FFMpegWriter(fps=20, bitrate=2400), dpi=110)
    plt.close(fig)

# ---------------- excess maps ----------------
acc = ld("acceptance.u8", np.uint8).reshape(4, NB).astype(bool)
mK0 = ld("rate_model_K0.f32").reshape(4, NB).astype(float)
mEX = ld("rate_model_example.f32").reshape(4, NB).astype(float)
cEX = ld("counts_example_s1_T180.f32").reshape(4, NB).astype(float)
cK0 = ld("counts_K0_s1_T180.f32").reshape(4, NB).astype(float)
cBL = ld("counts_blind_T180.f32").reshape(4, NB).astype(float)
edges = np.linspace(-1, 1, 33)
def hist(v, dI, m):
    H, _, _ = np.histogram2d(tx[m], ty[m], bins=[edges, edges], weights=v[m])
    return H
ce = cells[EX]
fig, axs = plt.subplots(4, 4, figsize=(15, 14))
for dI in range(4):
    m = acc[dI]
    E0 = hist(mK0[dI] * 180, dI, m); EX1 = hist(mEX[dI] * 180, dI, m)
    rel = np.where(E0 > 0, (EX1 / np.where(E0 > 0, E0, 1) - 1) * 100, np.nan)
    im = axs[0, dI].imshow(rel.T, origin="lower", extent=[-1, 1, -1, 1], cmap="RdBu", vmin=-np.nanmax(np.abs(rel)), vmax=np.nanmax(np.abs(rel)))
    plt.colorbar(im, ax=axs[0, dI], shrink=0.8, label="%")
    axs[0, dI].set_title(f"D{dI+1}: expected change (%)\nexample cell vs K0 (mean models)", fontsize=8)
    for row, (cnt, lab) in enumerate(((cEX, f"example cell data, seed 1"), (cK0, "K0 (no body) data, seed 1"), (cBL, "BLIND data (het seed 101)")), start=1):
        O = hist(np.where(cnt[dI] >= 0, cnt[dI], 0), dI, m)
        Eh = E0 * O.sum() / max(E0.sum(), 1e-9)   # global scale fitted per detector
        pull = np.where(Eh > 0, (O - Eh) / np.sqrt(np.where(Eh > 0, Eh, 1)), np.nan)
        im = axs[row, dI].imshow(pull.T, origin="lower", extent=[-1, 1, -1, 1], cmap="RdBu", vmin=-6, vmax=6)
        plt.colorbar(im, ax=axs[row, dI], shrink=0.8, label="pull")
        axs[row, dI].set_title(f"D{dI+1}: {lab}\n180 d, pull vs scaled K0 model", fontsize=8)
for a in axs.flat:
    a.set_xlabel("tan theta_x", fontsize=7); a.set_ylabel("tan theta_y", fontsize=7); a.tick_params(labelsize=6)
fig.suptitle(f"ADR-014 excess maps (synthetic). Example cell: depth {ce['d']} m, L {ce['L']} m, drho +{ce['drho']} at (0, 20, -{ce['d']}). "
             "Negative pull = muon deficit = excess density.\n0.0625 x 0.0625 tan bins; heterogeneity (1.5 % node s.d., 30 m lattice) is present in all data rows", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(FIG, "fig_muo_excess_maps.png"), dpi=120); plt.close(fig)

# ---------------- detectability map ----------------
depths = [40, 80, 120, 160]; Ls = [5, 10, 20, 40]; drs = [0.05, 0.25, 0.60]
C = rules["cells"]
def get(d, L, dr):
    for k, v in C.items():
        if v["d"] == d and v["L"] == L and abs(v["drho"] - dr) < 1e-6:
            return v
fig, axs = plt.subplots(2, 3, figsize=(15, 11))
for j, dr in enumerate(drs):
    for i, (key, lab) in enumerate((("Z", "Z_emp (8 seeds, heterogeneity)"), ("Znuis", "Asimov Z with scale nuisance (no heterogeneity)"))):
        Z = np.array([[get(d, L, dr)[key]["180"] for d in depths] for L in Ls])
        ax = axs[i, j]
        im = ax.imshow(np.clip(Z, 0.01, None), origin="lower", cmap="viridis", norm=matplotlib.colors.LogNorm(vmin=0.3, vmax=300))
        for a in range(4):
            for b in range(4):
                v = get(depths[b], Ls[a], dr)
                s = f"{Z[a, b]:.1f}"
                if i == 0:
                    s += "\n" + ("DET" if v["det_any"] else "-") + (f" {v['first_T']:.0f}d" if v["first_T"] and v["first_T"] <= 180 else "")
                ax.text(b, a, s, ha="center", va="center", fontsize=7, color="w" if Z[a, b] < 20 else "k")
        ax.set_xticks(range(4)); ax.set_xticklabels([f"{d}\n({200-d} m above)" for d in depths], fontsize=7)
        ax.set_yticks(range(4)); ax.set_yticklabels([str(L) for L in Ls])
        ax.set_xlabel("depth below surface d (m)"); ax.set_ylabel("cube side L (m)")
        ax.set_title(f"drho = +{dr:.2f}: {lab}, T = 180 d", fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle("ADR-014 detectability (synthetic). DET = Z_emp >= 5 and AUC >= 0.99 at some T <= 180 d (first such T shown)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.0)
fig.savefig(os.path.join(FIG, "fig_muo_detectability.png"), dpi=130); plt.close(fig)

# ---------------- exposure curves ----------------
Ts = [30, 45, 90, 180, 365]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for j, dr in enumerate(drs):
    ax = axs[j]
    for d, col in zip(depths, ["tab:blue", "tab:cyan", "tab:orange", "tab:purple"]):
        for L, ls in ((20, "-"), (40, "--")):
            v = get(d, L, dr)
            ax.plot(Ts, [max(v["Z"][str(T)], 0.05) for T in Ts], ls, color=col, marker="o", ms=3, label=f"d {d}, L {L}")
            if L == 20:
                ax.plot(Ts, [v["Zideal"][str(T)] for T in Ts], ":", color=col, lw=0.8)
    ax.axhline(5, color="k", lw=0.8); ax.axvline(180, color="0.5", lw=0.6)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.05, 500)
    ax.set_xlabel("exposure per detector (days)"); ax.set_title(f"drho = +{dr:.2f}  (solid L=20, dashed L=40; dotted = Asimov ideal, L=20)", fontsize=8)
axs[0].set_ylabel("Z_emp"); axs[2].legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_muo_curves.png"), dpi=130); plt.close(fig)

# ---------------- template scan slices ----------------
def scan_fig(fn, lo, n, truth, est, title, ax1, ax2):
    v = ld(fn).reshape(n[2], n[1], n[0])
    iz = int(round((est[2] - lo[2]) / 2)); iy = int(round((est[1] - lo[1]) / 2))
    xs = lo[0] + 2 * np.arange(n[0]); ys = lo[1] + 2 * np.arange(n[1]); zs = lo[2] + 2 * np.arange(n[2])
    a = v[:, iy, :]; b = v[iz, :, :]
    im1 = ax1.imshow(a, origin="lower", extent=[xs[0], xs[-1], zs[0], zs[-1]], aspect="auto", cmap="magma_r")
    plt.colorbar(im1, ax=ax1, shrink=0.8, label="dF (lower = better fit)")
    ax1.plot(truth[0], truth[2], "c+", ms=14, mew=2, label="truth"); ax1.plot(est[0], est[2], "wx", ms=10, mew=2, label="estimate")
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("z (m)"); ax1.set_title(title + f"\nx-z slice at y = {ys[iy]:.0f}", fontsize=8); ax1.legend(fontsize=7)
    im2 = ax2.imshow(b, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect="equal", cmap="magma_r")
    plt.colorbar(im2, ax=ax2, shrink=0.8, label="dF")
    ax2.plot(truth[0], truth[1], "c+", ms=14, mew=2); ax2.plot(est[0], est[1], "wx", ms=10, mew=2)
    ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)"); ax2.set_title(f"x-y slice at z = {zs[iz]:.0f} (dF, lower = better)", fontsize=8)
locs = list(csv.DictReader(open(os.path.join(D, "localisation.csv"))))
exr = [r for r in locs if int(r["cell"]) == EX and r["seed"] == "1"][0]
fig, axs = plt.subplots(2, 3, figsize=(16, 10))
scan_fig("scan_dF_example_s1.f32", (-60, -40, -190), (61, 61, 91), (0, 20, -float(ce["d"])),
         (float(exr["est_x"]), float(exr["est_y"]), float(exr["est_z"])), "example cell, seed 1, 180 d", axs[0, 0], axs[0, 1])
scan_fig("scan_dF_blind.f32", (-70, -70, -190), (71, 71, 91), BT, BE, f"BLIND (error {float(bt['error_m'][0]):.2f} m)", axs[1, 0], axs[1, 1])
# localisation error per cell (8 seeds)
ax = axs[0, 2]
xs_ = []; lab = []
for i, c in enumerate(sorted(cells)):
    e = [float(r["dist_m"]) for r in locs if int(r["cell"]) == c]
    col = "tab:green" if rules["cells"][str(c)]["det180"] else "0.6"
    ax.scatter([i] * len(e), e, s=6, color=col)
    tol = max(5, float(cells[c]["L"]) / 2)
    ax.plot([i - 0.4, i + 0.4], [tol, tol], color="k", lw=0.6)
ax.set_yscale("log"); ax.set_xlabel("cell (d, L, drho order; see cells.csv)"); ax.set_ylabel("localisation error (m)")
ax.set_title("template-scan error, 8 seeds at 180 d (green: detected at 180 d; bar = tolerance)", fontsize=8)
axs[1, 2].axis("off")
axs[1, 2].text(0, 0.95, "\n".join([f"R{k[1]}: {'PASS' if rules[k]['pass_'] else 'FAIL'}" for k in ("R1", "R2", "R3", "R4", "R5", "R6")]),
               va="top", family="monospace", fontsize=11)
fig.suptitle("ADR-014 template-scan localisation (synthetic; uses data + K0 model + hypothesis L, drho only)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(os.path.join(FIG, "fig_muo_scan.png"), dpi=120); plt.close(fig)

# ---------------- MLEM recon slices ----------------
if os.path.exists(os.path.join(D, "recon2m_example_s1.f32")):
    rnx, rny, rnz = 240, 210, 102
    rec = ld("recon2m_example_s1.f32").reshape(rnz, rny, rnx)
    tru = ld("truth2m_example.f32").reshape(rnz, rny, rnx)
    dom = ld("domain2m.u8", np.uint8).reshape(rnz, rny, rnx).astype(bool)
    X = -240 + 2 * (np.arange(rnx) + 0.5); Y = -210 + 2 * (np.arange(rny) + 0.5); Z = -204 + 2 * (np.arange(rnz) + 0.5)
    cz = -float(ce["d"]); ix = np.argmin(abs(X - 0)); iy = np.argmin(abs(Y - 20)); iz = np.argmin(abs(Z - cz))
    xm = (X >= -60) & (X <= 60); ym = (Y >= -40) & (Y <= 80); zm = (Z >= -190) & (Z <= -10)
    def box5(v):
        # 5x5x5 voxel (10 m) moving average, display only
        out = v.astype(np.float64)
        for ax_ in range(3):
            c = np.cumsum(np.pad(out, [(3, 2) if a == ax_ else (0, 0) for a in range(3)], mode="edge"), axis=ax_)
            sl_hi = [slice(None)] * 3; sl_lo = [slice(None)] * 3
            sl_hi[ax_] = slice(5, None); sl_lo[ax_] = slice(0, -5)
            out = (c[tuple(sl_hi)] - c[tuple(sl_lo)]) / 5.0
        return out
    recs = np.where(dom, box5(np.where(dom, rec, 2.70)), rec)
    fig, axs = plt.subplots(3, 3, figsize=(15, 13))
    for r, (vol, nm) in enumerate(((tru, "truth"), (rec, "MLEM recon"), (recs, "MLEM recon, 10 m box-smoothed (display only)"))):
        sl = [(vol[:, iy, :][np.ix_(zm, xm)], [X[xm][0], X[xm][-1], Z[zm][0], Z[zm][-1]], "x", "z", f"x-z at y = {Y[iy]:.0f}"),
              (vol[:, :, ix][np.ix_(zm, ym)], [Y[ym][0], Y[ym][-1], Z[zm][0], Z[zm][-1]], "y", "z", f"y-z at x = {X[ix]:.0f}"),
              (vol[iz, :, :][np.ix_(ym, xm)], [X[xm][0], X[xm][-1], Y[ym][0], Y[ym][-1]], "x", "y", f"x-y at z = {Z[iz]:.0f}")]
        for c, (img, ext, xl, yl, t) in enumerate(sl):
            im = axs[r, c].imshow(img, origin="lower", extent=ext, cmap="cividis", vmin=2.55, vmax=3.35, aspect="equal")
            axs[r, c].set_title(f"{nm}: {t}", fontsize=8); axs[r, c].set_xlabel(xl + " (m)"); axs[r, c].set_ylabel(yl + " (m)")
            plt.colorbar(im, ax=axs[r, c], shrink=0.8, label="g/cm$^3$")
    fig.suptitle(f"ADR-014 MLEM density reconstruction (figure only; 4 detectors, 180 d, seed 1): cube L {ce['L']} m, +{ce['drho']} at depth {ce['d']} m\n"
                 "2 m grid, 20 iterations, domain x[-60,60] y[-40,80] z[-190,-10]; prior = mean host 2.70", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(os.path.join(FIG, "fig_muo_recon_slices.png"), dpi=120); plt.close(fig)
print(sorted(os.listdir(FIG)))
