"""ADR-013: dwave-neal baseline on the QUBO files written by check_ree_qubo.
Usage: python tools/ree_neal.py <dir> [num_sweeps=420] [reads=64]
Writes neal_runs.csv: id, read, energy, feasible, cost, hit, time_us (per read = total/reads)."""
import sys, glob, os, time, csv
import numpy as np
import dimod, neal

d = sys.argv[1]
sweeps = int(sys.argv[2]) if len(sys.argv) > 2 else 420
reads = int(sys.argv[3]) if len(sys.argv) > 3 else 64
sampler = neal.SimulatedAnnealingSampler()
rows = []
files = sorted(glob.glob(os.path.join(d, 'qubo_*.txt')))
for f in files:
    iid = os.path.basename(f)[5:-4]
    L = open(f).read().split('\n')
    nv, nq, offset, P, opt = L[0].split(); nv, nq = int(nv), int(nq); offset, P, opt = float(offset), float(P), float(opt)
    diag = np.zeros(nv); cn = np.zeros(nv); Q = {}
    for i in range(nv):
        a, qd, c = L[1 + i].split(); diag[int(a)] = float(qd); cn[int(a)] = float(c)
    for k in range(nq):
        i, j, v = L[1 + nv + k].split(); Q[(int(i), int(j))] = float(v)
    bqm = dimod.BinaryQuadraticModel({i: diag[i] for i in range(nv)}, Q, offset, dimod.BINARY)
    t0 = time.perf_counter()
    ss = sampler.sample(bqm, num_reads=reads, num_sweeps=sweeps, seed=12345)
    dt = (time.perf_counter() - t0) / reads * 1e6
    X = ss.record.sample; E = ss.record.energy
    for r in range(len(E)):
        x = X[r].astype(float)
        cost_n = float(x @ cn)
        e = float(offset + x @ diag + sum(v * x[i] * x[j] for (i, j), v in Q.items()))
        feas = abs(e - cost_n) < 1e-9
        cost = cost_n * P
        hit = feas and abs(cost - opt) <= 1e-9 * opt
        rows.append(dict(id=iid, read=r + 1, energy=e, feasible=int(feas), cost=cost, hit=int(hit), time_us=dt))
    print(iid, 'hits', sum(r['hit'] for r in rows if r['id'] == iid), '/', reads, f'{dt:.0f} us/read', flush=True)
with open(os.path.join(d, 'neal_runs.csv'), 'w', newline='') as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('neal', neal.__version__, 'dimod', dimod.__version__, 'sweeps', sweeps, 'reads', reads)
