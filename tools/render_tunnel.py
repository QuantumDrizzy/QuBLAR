"""ADR-012 figures. Usage: python tools/render_tunnel.py <experiments/tunnel>"""
import sys, os, csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

D = sys.argv[1]
FIG = os.path.join(D, 'figures'); os.makedirs(FIG, exist_ok=True)
VX, VY, VZ = 100, 50, 50
Vs = ['inf', '50', '30', '20', '15', '10', '5']

def load_cells(d):
    return list(csv.DictReader(open(os.path.join(d, 'tunnel_cells.csv'))))

def mips(d):
    idx = list(csv.DictReader(open(os.path.join(d, 'tunnel_mips_index.csv'))))
    raw = np.fromfile(os.path.join(d, 'tunnel_mips.f32'), np.float32)
    rec = VZ * VX + VY * VX
    out = {}
    for r in idx:
        i = int(r['i']); a = raw[i * rec:(i + 1) * rec]
        out[(r['sensor'], r['V'], r['B'], r['pipe'], r['scene'], r['noisy'])] = (a[:VZ * VX].reshape(VZ, VX), a[VZ * VX:].reshape(VY, VX))
    return out

tg = list(csv.DictReader(open(os.path.join(D, 'tunnel_targets.csv'))))
t1 = {r['scene']: r for r in tg if r['seed'] == '1'}

# ---------------------------------------------------------------- 1. scene render
def box_faces(x0, x1, y0, y1, z0, z1):
    P = lambda x, y, z: (x, z, y)   # plot (x, z, y) so 'up' is vertical
    return [[P(x0, y0, z0), P(x1, y0, z0), P(x1, y1, z0), P(x0, y1, z0)],
            [P(x0, y0, z1), P(x1, y0, z1), P(x1, y1, z1), P(x0, y1, z1)],
            [P(x0, y0, z0), P(x0, y0, z1), P(x0, y1, z1), P(x0, y1, z0)],
            [P(x1, y0, z0), P(x1, y0, z1), P(x1, y1, z1), P(x1, y1, z0)],
            [P(x0, y0, z0), P(x1, y0, z0), P(x1, y0, z1), P(x0, y0, z1)],
            [P(x0, y1, z0), P(x1, y1, z0), P(x1, y1, z1), P(x0, y1, z1)]]

fig = plt.figure(figsize=(12, 7)); ax = fig.add_subplot(111, projection='3d')
P = lambda x, y, z: (x, z, y)
walls = [
    [P(-2.5, 0, -10), P(2.5, 0, -10), P(2.5, 0, 6), P(-2.5, 0, 6)],
    [P(2.5, 0, 1), P(22.5, 0, 1), P(22.5, 0, 6), P(2.5, 0, 6)],
    [P(-2.5, 0, -10), P(-2.5, 0, 6), P(-2.5, 5, 6), P(-2.5, 5, -10)],
    [P(2.5, 0, -10), P(2.5, 5, -10), P(2.5, 5, 1), P(2.5, 0, 1)],
    [P(-2.5, 0, 6), P(22.5, 0, 6), P(22.5, 5, 6), P(-2.5, 5, 6)],
    [P(2.5, 0, 1), P(2.5, 5, 1), P(22.5, 5, 1), P(22.5, 0, 1)],
    [P(22.5, 0, 1), P(22.5, 5, 1), P(22.5, 5, 6), P(22.5, 0, 6)]]
ax.add_collection3d(Poly3DCollection(walls, facecolor=(0.45, 0.4, 0.35, 0.12), edgecolor=(0.3, 0.3, 0.3, 0.5), lw=0.5))
ax.add_collection3d(Poly3DCollection([[P(1, 0.5, 5.99), P(5, 0.5, 5.99), P(5, 4.5, 5.99), P(1, 4.5, 5.99)]], facecolor=(0.9, 0.7, 0.1, 0.45)))
pc = t1['person']; vc = t1['vehicle']
cx, cz = float(pc['cx']), float(pc['cz'])
for b in [(cx - .175, cx - .025, 0, .85, cz - .1, cz + .1), (cx + .025, cx + .175, 0, .85, cz - .1, cz + .1),
          (cx - .225, cx + .225, .85, 1.45, cz - .15, cz + .15), (cx - .1, cx + .1, 1.45, 1.7, cz - .1, cz + .1)]:
    ax.add_collection3d(Poly3DCollection(box_faces(*b), facecolor=(0.85, 0.2, 0.2, 0.9)))
vx_, vz_ = float(vc['cx']), float(vc['cz'])
for b in [(vx_ - 1.25, vx_ + 1.25, .3, 1.3, vz_ - .75, vz_ + .75), (vx_ - 1.25, vx_ - .25, 1.3, 1.8, vz_ - .75, vz_ + .75)]:
    ax.add_collection3d(Poly3DCollection(box_faces(*b), facecolor=(0.2, 0.4, 0.85, 0.55)))
for s, lab, col in [((-1.5, 1.6, -4), 'sensor (far)', 'k'), ((0, 1.6, 3), 'sensor (near, variant)', 'gray')]:
    ax.scatter(*P(*s), color=col, s=40); ax.text(*P(s[0], s[1] + 0.4, s[2]), lab, fontsize=8)
    for c in [(1, .5, 6), (5, .5, 6), (5, 4.5, 6), (1, 4.5, 6)]:
        ax.plot(*zip(P(*s), P(*c)), color=col, lw=0.6, alpha=0.6)
ax.text(*P(3, 4.8, 6), 'relay patch 4x4 m (dark rock, rho 0.1-0.3)', fontsize=8)
ax.text(*P(cx, 2.1, cz), 'person 1.7 m', fontsize=8, color='darkred')
ax.text(*P(vx_, 2.3, vz_), 'vehicle', fontsize=8, color='navy')
ax.set_xlim(-3, 16); ax.set_ylim(-10, 7); ax.set_zlim(0, 6)
ax.set_box_aspect((19, 17, 6)); ax.view_init(elev=38, azim=-62)
ax.set_xlabel('x (m)'); ax.set_ylabel('z (m, along access drift)'); ax.set_zlabel('y (m)')
ax.set_title('ADR-012 scene: L-bend drift, relay patch on end wall, hidden person / vehicle in side drift (seed 1)')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_tunnel_scene.png'), dpi=140); plt.close()

# ---------------------------------------------------------------- 2. volumes
M0 = mips(D); M4 = mips(os.path.join(D, 'exploratory_budget_x1e4'))
ext_top = [4, 14, 1, 6]
cols = [('clean P2 (expected, no noise)', M0, ('far', 'inf', '1e+12', 'P2', None, '0')),
        ('P2, B=1e12, V=inf (frozen grid)', M0, ('far', 'inf', '1e+12', 'P2', None, '1')),
        ('P2 whitened, B=1e12, V=inf', M0, ('far', 'inf', '1e+12', 'P2', None, '2')),
        ('P2, B=1e17, V=inf (exploratory)', M4, ('far', 'inf', '1e+17', 'P2', None, '1')),
        ('P2, B=1e17, V=20 m (exploratory)', M4, ('far', '20', '1e+17', 'P2', None, '1'))]
fig, axs = plt.subplots(3, len(cols), figsize=(4.2 * len(cols), 7.2))
for i, scn in enumerate(['person', 'vehicle', 'empty']):
    for j, (title, M, k) in enumerate(cols):
        kk = k[:4] + (scn,) + k[5:]
        ax = axs[i, j]
        if kk not in M: ax.axis('off'); continue
        top = M[kk][0]
        ax.imshow(top, origin='lower', extent=ext_top, cmap='inferno', aspect='equal')
        if scn in t1:
            r = t1[scn]
            ax.add_patch(plt.Rectangle((float(r['lo_x']), float(r['lo_z'])), float(r['hi_x']) - float(r['lo_x']),
                                       float(r['hi_z']) - float(r['lo_z']), fill=False, ec='cyan', lw=1))
        if i == 0: ax.set_title(title, fontsize=9)
        if j == 0: ax.set_ylabel(f'{scn}\nz (m)')
        ax.set_xlabel('x (m)', fontsize=8)
fig.suptitle('Backprojected volumes, max projection over height (top view of the side drift). Cyan = true target box. Seed 1, far sensor.')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_tunnel_volumes.png'), dpi=130); plt.close()

# ---------------------------------------------------------------- 3. detection maps (merged budget axis)
C0 = load_cells(D); C2 = load_cells(os.path.join(D, 'exploratory_budget_x100')); C4 = load_cells(os.path.join(D, 'exploratory_budget_x1e4'))
def pick(sn, V, B, pipe, tgt):
    b = float(B)
    src = C0 if b <= 1e13 else (C2 if b <= 1e15 else C4)
    return next(c for c in src if c['sensor'] == sn and c['V'] == V and float(c['B']) == b and c['pipe'] == pipe and c['target'] == tgt)
Bax = [1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17]
for stat, zk, dk, lk, tag in [('frozen T', 'Z', 'detected', 'localised', 'frozen'), ('whitened T (exploratory)', 'Zw', 'detected_w', 'localised_w', 'whitened')]:
    fig, axs = plt.subplots(2, 4, figsize=(20, 8.5))
    for i, pipe in enumerate(['P2', 'P1']):
        for j, (sn, tgt) in enumerate([('far', 'person'), ('far', 'vehicle'), ('near', 'person'), ('near', 'vehicle')]):
            Z = np.zeros((len(Vs), len(Bax))); ok = np.zeros_like(Z, bool)
            for a, V in enumerate(Vs):
                for b, B in enumerate(Bax):
                    c = pick(sn, V, B, pipe, tgt); Z[a, b] = float(c[zk]); ok[a, b] = c[dk] == 'True' and c[lk] == 'True'
            ax = axs[i, j]
            im = ax.imshow(np.clip(Z, -2, 20), cmap='viridis', vmin=-2, vmax=20, aspect='auto')
            for a in range(len(Vs)):
                for b in range(len(Bax)):
                    ax.text(b, a, f"{Z[a, b]:.0f}" + ('\n✓' if ok[a, b] else ''), ha='center', va='center', fontsize=7,
                            color='k' if Z[a, b] > 10 else 'w')
            ax.axvline(3.5, color='r', lw=2); ax.text(3.6, -0.7, 'exploratory ->', color='r', fontsize=8)
            ax.set_xticks(range(len(Bax))); ax.set_xticklabels([f"1e{int(np.log10(b))}" for b in Bax], fontsize=8)
            ax.set_yticks(range(len(Vs))); ax.set_yticklabels([f"V={v}" for v in Vs], fontsize=8)
            ax.set_title(f"{pipe} {sn} sensor, {tgt}", fontsize=10)
            if i == 1: ax.set_xlabel('budget B (first-bounce photons / relay point, rho=1)')
    fig.colorbar(im, ax=axs, shrink=0.6, label='Z_emp')
    fig.suptitle(f"Detection map, {stat}: Z_emp per cell (✓ = detected AND localised). Frozen grid B<=1e13 left of red line.")
    plt.savefig(os.path.join(FIG, f'fig_tunnel_detection_map_{tag}.png'), dpi=120, bbox_inches='tight'); plt.close()

# ---------------------------------------------------------------- 4. curves
fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
for V, col in zip(Vs, plt.cm.plasma(np.linspace(0, 0.9, len(Vs)))):
    axs[0].plot(Bax, [float(pick('far', V, B, 'P2', 'person')['Z']) for B in Bax], 'o-', color=col, label=f"V={V} m")
    axs[1].plot(Bax, [float(pick('far', V, B, 'P2', 'person')['med_dist']) for B in Bax], 'o-', color=col)
axs[0].axhline(5, color='k', ls='--'); axs[0].axvline(1e13, color='r', lw=1); axs[1].axhline(0.5, color='k', ls='--'); axs[1].axvline(1e13, color='r', lw=1)
for a in axs[:2]: a.set_xscale('log'); a.set_xlabel('budget B')
axs[0].set_yscale('symlog'); axs[0].set_ylabel('Z_emp (frozen T), person, far, P2'); axs[0].legend(fontsize=7)
axs[1].set_ylabel('median distance peak -> target box (m)')
ph = list(csv.DictReader(open(os.path.join(D, 'tunnel_photons.csv'))))
for tgt, mk in [('person', 'o'), ('vehicle', 's')]:
    for sn, ls in [('far', '-'), ('near', '--')]:
        y = [np.median([float(r['net_vs_empty']) for r in ph if r['sensor'] == sn and r['V'] == V and r['B'] == '1e+12' and r['scene'] == tgt]) for V in Vs]
        axs[2].plot(range(len(Vs)), np.abs(y), mk + ls, label=f"{tgt}, {sn}")
yd = [np.median([float(r['dust_photons']) for r in ph if r['sensor'] == 'far' and r['V'] == V and r['B'] == '1e+12' and r['scene'] == 'empty']) for V in Vs]
axs[2].plot(range(len(Vs)), np.maximum(yd, 1e-12), 'k:', label='dust backscatter (far)')
axs[2].set_yscale('log'); axs[2].set_xticks(range(len(Vs))); axs[2].set_xticklabels([f"{v}" for v in Vs]); axs[2].set_xlabel('visibility V (m)')
axs[2].set_ylabel('expected photons per 1024-point scan at B=1e12'); axs[2].legend(fontsize=7)
axs[2].set_title('net target photons |scene - empty|')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_tunnel_curves.png'), dpi=130); plt.close()

# ---------------------------------------------------------------- 5. waveforms
meta = dict(l.strip().split('=') for l in open(os.path.join(D, 'tunnel_meta.txt')) if '=' in l)
t0 = float(meta['t0_ns']); tb = np.arange(1280) * 0.1 + t0
we = np.fromfile(os.path.join(D, 'tunnel_wave_empty.f32'), np.float32).reshape(7, 1280)
wp = np.fromfile(os.path.join(D, 'tunnel_wave_person.f32'), np.float32).reshape(7, 1280)
fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
for l, col in zip([0, 1, 3, 5], ['k', 'b', 'g', 'r']):
    axs[0].semilogy(tb, np.maximum(we[l] * 1e12, 1e-6), color=col, label=f"empty, V={Vs[l]}")
    axs[1].plot(tb, (wp[l] - we[l]) * 1e12, color=col, label=f"person - empty, V={Vs[l]}")
axs[0].set_ylim(1e-4, None); axs[0].set_xlabel('time (ns)'); axs[0].set_ylabel('expected counts / 100 ps bin at B=1e12'); axs[0].legend(fontsize=8)
axs[0].set_title('centre relay point: tunnel clutter + dust backscatter')
axs[1].set_xlabel('time (ns)'); axs[1].set_ylabel('expected counts / bin'); axs[1].legend(fontsize=8); axs[1].set_title('the hidden-person signal at the same relay point')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_tunnel_waveforms.png'), dpi=130); plt.close()

# ---------------------------------------------------------------- 6. ghost figures
for sub, tag in [('ghost', 'frozen'), (os.path.join('ghost', 'exploratory_return_only'), 'return_only_exploratory')]:
    gd = os.path.join(D, sub)
    if not os.path.exists(os.path.join(gd, 'ghost_results.csv')): continue
    G = list(csv.DictReader(open(os.path.join(gd, 'ghost_results.csv'))))
    I = np.load(os.path.join(gd, 'ghost_images_seed1.npz'))
    taus = sorted({float(r['tau']) for r in G})
    meths = ['camera', 'ghost', 'camera_wiener', 'ghost_wiener']
    fig, axs = plt.subplots(len(meths), len(taus) + 1, figsize=(2.0 * (len(taus) + 1), 2.1 * len(meths)))
    for i, m in enumerate(meths):
        axs[i, 0].imshow(I['truth'], cmap='gray', vmin=0, vmax=1); axs[i, 0].set_title('truth' if i == 0 else '', fontsize=8)
        axs[i, 0].set_ylabel(m, fontsize=9)
        for j, t in enumerate(taus):
            key = f"{m}|1e+07|{t:g}" if f"{m}|1e+07|{t:g}" in I.files else f"{m}|1e+07|{t}"
            im = I[key]
            p = [float(r['psnr']) for r in G if r['method'] == m and float(r['tau']) == t and float(r['N']) == 1e7 and r['seed'] == '1'][0]
            axs[i, j + 1].imshow(np.clip(im, 0, 1), cmap='gray', vmin=0, vmax=1)
            axs[i, j + 1].set_title((f"OD {t:g}\n" if i == 0 else '') + f"{p:.1f} dB", fontsize=8)
    for a in axs.ravel(): a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Ghost imaging vs camera through dust, N = 1e7 photons, seed 1 ({tag})", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, f'fig_ghost_grid_{tag}.png'), dpi=130); plt.close()
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    for j, N in enumerate([1e5, 1e7, 1e9]):
        for m, col in zip(meths, ['C0', 'C3', 'C0', 'C3']):
            ls = '-' if 'wiener' not in m else '--'
            for i, met in enumerate(['psnr', 'ssim']):
                vals = np.array([[float(r[met]) for r in G if r['method'] == m and float(r['tau']) == t and float(r['N']) == N] for t in taus])
                axs[i, j].plot(taus, np.median(vals, 1), ls, color=col, marker='o', ms=3, label=m)
                axs[i, j].fill_between(taus, vals.min(1), vals.max(1), color=col, alpha=0.15)
        axs[0, j].set_title(f"N = {N:.0e} photons"); axs[1, j].set_xlabel('one-way optical depth tau')
        axs[0, j].set_ylabel('PSNR (dB)'); axs[1, j].set_ylabel('SSIM')
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"PSNR / SSIM vs optical depth, median and range over 8 seeds ({tag})")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, f'fig_ghost_curves_{tag}.png'), dpi=130); plt.close()
print('figures in', FIG, sorted(os.listdir(FIG)))
