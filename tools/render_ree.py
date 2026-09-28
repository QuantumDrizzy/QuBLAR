"""ADR-013 figures. Usage: python tools/render_ree.py <experiments/ree_cascade>"""
import sys, os, csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = sys.argv[1]
FIG = os.path.join(D, 'figures'); os.makedirs(FIG, exist_ok=True)
rd = lambda d, f: list(csv.DictReader(open(os.path.join(d, f))))
inst = {r['id']: r for r in rd(D, 'instances.csv')}
EXP = os.path.join(D, 'exploratory_long_schedule')

# ---------------------------------------------------------------- flowsheets
def flowsheet(iid, ax):
    I = inst[iid]; n = int(I['n']); names = ['La', 'Ce', 'Pr', 'Nd', 'Sm+'][:n]
    z = [float(v) for v in I['z'].split(';')]; beta = [float(v) for v in I['beta'].split(';')]
    N = [int(v) for v in I['stages'].split(';')]
    splits = {}
    for t in I['opt_tree'].split(';'):
        a, s, b = map(int, t.split('-')); splits[(a, b)] = s
    pos = {}
    def place(a, b, depth):
        if (a, b) in splits:
            s = splits[(a, b)]
            place(a, s, depth + 1); place(s + 1, b, depth + 1)
            pos[(a, b)] = ((pos[(a, s)][0] + pos[(s + 1, b)][0]) / 2, -depth)
        else:
            pos[(a, b)] = (a * 10.0 / max(1, n - 1), -depth)
    place(0, n - 1, 0)
    for (a, b), (x, y) in pos.items():
        lab = ' '.join(names[a:b + 1]); F = sum(z[a:b + 1])
        if (a, b) in splits:
            s = splits[(a, b)]
            txt = f"SX: {' '.join(names[a:s + 1])} | {' '.join(names[s + 1:b + 1])}\nbeta {beta[s]:.2f}, N = {N[s]} stages\nF_in = {F:.3f}, cost {F * N[s]:.2f}"
            ax.text(x, y, txt, ha='center', va='center', fontsize=8, bbox=dict(boxstyle='round', fc='#dde8f5', ec='#335'))
            for ch in [(a, s), (s + 1, b)]:
                cx, cy = pos[ch]; ax.annotate('', xy=(cx, cy + 0.28), xytext=(x, y - 0.3), arrowprops=dict(arrowstyle='->', color='#444'))
            ax.text((x + pos[(a, s)][0]) / 2 - 0.2, y - 0.5, 'raffinate', fontsize=7, color='#666', ha='right')
            ax.text((x + pos[(s + 1, b)][0]) / 2 + 0.2, y - 0.5, 'extract', fontsize=7, color='#666')
        else:
            ax.text(x, y, f"{lab}\n{100 * F:.2f} mol%", ha='center', va='center', fontsize=9, bbox=dict(boxstyle='round', fc='#e6f4e0', ec='#363'))
    ax.text(5, 0.7, f"feed (mol fraction): " + ', '.join(f"{nm} {v:.3f}" for nm, v in zip(names, z)), ha='center', fontsize=8)
    ax.set_xlim(-0.5, 10.5); ax.set_ylim(-n + 0.4, 1.1); ax.axis('off')
    ax.set_title(f"{iid}: optimal sequence (DP-proven), total flow x stages = {float(I['opt']):.2f}; greedy {float(I['greedy']):.2f}", fontsize=10)

fig, axs = plt.subplots(1, 2, figsize=(18, 7))
flowsheet('MP_P507-A', axs[0]); flowsheet('MP_P507-B', axs[1])
fig.suptitle('Mountain Pass bastnaesite feed (Haxel 2005, Table 4) -- light-REE SX separation sequence', fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_ree_flowsheet.png'), dpi=130); plt.close()

# ---------------------------------------------------------------- landscape
H = rd(D, 'landscape_hist_MP_P507-A.csv'); Fz = rd(D, 'landscape_feasible_MP_P507-A.csv')
lo = np.array([float(r['bin_lo']) for r in H]); cnt = np.array([int(r['count']) for r in H])
fe = np.array([float(r['energy']) for r in Fz]); P = float(inst['MP_P507-A']['P']); opt = float(inst['MP_P507-A']['opt'])
fig, axs = plt.subplots(1, 2, figsize=(15, 4.8))
axs[0].bar(lo, np.maximum(cnt, 0.8), width=0.1, align='edge', color='gray', log=True)
axs[0].set_xlim(0, 20); axs[0].set_xlabel('QUBO energy (units of the penalty P)'); axs[0].set_ylabel('number of states (of 2^20)')
for e in fe: axs[0].axvline(e, color='C0', lw=0.8)
axs[0].axvline(opt / P, color='C3', lw=2, label='optimum (feasible)'); axs[0].legend()
axs[0].set_title('MP_P507-A: energy landscape of all 1,048,576 states; blue = the 14 valid sequences')
axs[1].plot(sorted(fe * P), 'o-'); axs[1].axhline(opt, color='C3', ls='--')
axs[1].set_xlabel('valid sequence (sorted)'); axs[1].set_ylabel('cost = sum flow x stages'); axs[1].set_title('the 14 valid sequences: cost spread')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_ree_landscape.png'), dpi=130); plt.close()

# ---------------------------------------------------------------- convergence
fig, axs = plt.subplots(1, 2, figsize=(15, 4.8))
for j, iid in enumerate(['MP_P507-A', 'MP_P507-B']):
    T = rd(D, f'energy_trace_{iid}.csv')
    for sd in range(1, 9):
        e = [float(r['energy']) for r in T if int(r['seed']) == sd]
        axs[j].plot(e, lw=0.9, label=f'seed {sd}')
    axs[j].axhline(float(inst[iid]['opt']) / float(inst[iid]['P']), color='k', ls='--', label='optimum')
    axs[j].set_yscale('symlog', linthresh=1); axs[j].set_xlabel('sweep'); axs[j].set_ylabel('QUBO energy (units of P)')
    tw = axs[j].twinx(); sw = np.arange(420); temp = np.where(sw < 400, 5 * (1e-3 / 5) ** (np.minimum(sw, 399) / 399), 1e-3)
    tw.semilogy(sw, temp, 'k:', lw=1); tw.set_ylabel('temperature (dotted)')
    axs[j].set_title(f'{iid}: engine anneal traces (Gset MAP schedule)'); axs[j].legend(fontsize=7, ncol=3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_ree_convergence.png'), dpi=130); plt.close()

# ---------------------------------------------------------------- solver comparison
R = json.load(open(os.path.join(D, 'ree_rules.json')))
RX = json.load(open(os.path.join(EXP, 'ree_rules.json'))) if os.path.exists(os.path.join(EXP, 'ree_rules.json')) else None
ns = sorted(int(k) for k in R['per_n'])
fig, axs = plt.subplots(1, 4, figsize=(21, 4.6))
for rr, ls, tag in [(R, '-', '420 sweeps (frozen)'), (RX, '--', '4200 sweeps (exploratory)')]:
    if rr is None: continue
    g = lambda k: [rr['per_n'][str(n)][k] for n in ns]
    axs[0].plot(ns, np.array(g('eng_hit8')) / np.array(g('k')), 'o' + ls, color='C3', label=f'engine {tag}')
    axs[0].plot(ns, np.array(g('neal_hit8')) / np.array(g('k')), 's' + ls, color='C0', label=f'neal {tag}')
    axs[1].plot(ns, g('eng_ps_mean'), 'o' + ls, color='C3'); axs[1].plot(ns, g('neal_ps_mean'), 's' + ls, color='C0')
    axs[2].plot(ns, g('eng_feas_mean'), 'o' + ls, color='C3'); axs[2].plot(ns, g('neal_feas_mean'), 's' + ls, color='C0')
    axs[3].plot(ns, g('eng_t_us'), 'o' + ls, color='C3'); axs[3].plot(ns, g('neal_t_us'), 's' + ls, color='C0')
axs[0].set_ylabel('fraction of instances: best-of-8 = optimum'); axs[1].set_ylabel('mean p_s (single run hits optimum)')
axs[2].set_ylabel('mean fraction of runs ending feasible'); axs[3].set_ylabel('wall time per run (us)'); axs[3].set_yscale('log')
for a in axs: a.set_xlabel('number of components n (vars = n(n^2-1)/6)')
axs[0].legend(fontsize=7)
fig.suptitle('Engine (anneal_branch, pair-ray encoding) vs neal on the sequencing QUBO; 5 = includes the 2 real instances')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_ree_solvers.png'), dpi=130); plt.close()

S = rd(D, 'ree_summary.csv')
fig, ax = plt.subplots(figsize=(7, 6))
def fz(v):
    v = float(v); return 0.6 if not np.isfinite(v) else v
x = [fz(r['neal_gap8_bestfeas']) for r in S]; y = [fz(r['eng_gap8_bestfeas']) for r in S]
c = [int(r['n']) for r in S]
sc = ax.scatter(x, y, c=c, cmap='viridis'); plt.colorbar(sc, label='n')
ax.plot([0, 0.6], [0, 0.6], 'k--', lw=0.8); ax.set_xlabel('neal best feasible of 8: gap to optimum'); ax.set_ylabel('engine best feasible of 8: gap')
ax.set_title('per-instance optimality gap (0.6 = no feasible run in 8)')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_ree_gap_scatter.png'), dpi=130); plt.close()
print(sorted(os.listdir(FIG)))
