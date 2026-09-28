"""ADR-013 rule evaluation (R1-R5). Usage: python tools/ree_analyze.py <dir>
Reads instances.csv, engine_runs.csv, neal_runs.csv, brute_force.csv, energy_check.csv;
writes ree_summary.csv and ree_rules.json."""
import sys, os, csv, json, math
import numpy as np

d = sys.argv[1]
rd = lambda f: list(csv.DictReader(open(os.path.join(d, f))))
inst = rd('instances.csv'); eng = rd('engine_runs.csv'); nea = rd('neal_runs.csv')
bf = rd('brute_force.csv'); ec = rd('energy_check.csv')

def solver_stats(runs, key, opt, first8):
    rs = sorted(runs, key=lambda r: int(r[key]))
    e = [float(r['energy']) for r in rs]; feas = [r['feasible'] == '1' for r in rs]
    cost = [float(r['cost']) for r in rs]; hit = [r['hit'] == '1' for r in rs]
    t = np.mean([float(r['time_us']) for r in rs]) * 1e-6
    b8 = int(np.argmin(e[:first8]))
    gap8 = (cost[b8] - opt) / opt if feas[b8] else math.inf
    feas8 = [i for i in range(first8) if feas[i]]
    gap8_feas = min((cost[i] - opt) / opt for i in feas8) if feas8 else math.inf
    ps = sum(hit) / len(hit)
    tts = t if ps >= 1 else (math.inf if ps == 0 else t * math.log(0.01) / math.log(1 - ps))
    return dict(gap8=gap8, gap8_bestfeasible=gap8_feas, hit8=gap8 == 0 or (feas[b8] and abs(cost[b8] - opt) <= 1e-9 * opt),
                ps=ps, feas_frac=sum(feas) / len(feas), t_run=t, tts99=tts)

rows = []
for I in inst:
    iid, opt = I['id'], float(I['opt'])
    E = solver_stats([r for r in eng if r['id'] == iid], 'seed', opt, 8)
    N = solver_stats([r for r in nea if r['id'] == iid], 'read', opt, 8)
    g = float(I['greedy'])
    # R3 per-instance outcome
    if E['gap8'] < N['gap8'] - 1e-12: res = 'win'
    elif E['gap8'] > N['gap8'] + 1e-12: res = 'loss'
    elif (E['gap8'] == N['gap8']) or (math.isinf(E['gap8']) and math.isinf(N['gap8'])) or abs(E['gap8'] - N['gap8']) <= 1e-12:
        dp = E['ps'] - N['ps']
        res = 'tie' if abs(dp) <= 0.10 + 1e-12 else ('win' if dp > 0 else 'loss')
    rows.append(dict(id=iid, set=I['set'], n=int(I['n']), nvars=int(I['nvars']), opt=opt, greedy=g,
                     greedy_gap=(g - opt) / opt,
                     eng_gap8=E['gap8'], eng_gap8_bestfeas=E['gap8_bestfeasible'], eng_hit8=E['hit8'], eng_ps=E['ps'],
                     eng_feas=E['feas_frac'], eng_t_us=E['t_run'] * 1e6, eng_tts99_s=E['tts99'],
                     neal_gap8=N['gap8'], neal_gap8_bestfeas=N['gap8_bestfeasible'], neal_hit8=N['hit8'], neal_ps=N['ps'],
                     neal_feas=N['feas_frac'], neal_t_us=N['t_run'] * 1e6, neal_tts99_s=N['tts99'], r3=res))
with open(os.path.join(d, 'ree_summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def binom_two_sided(k, n):
    if n == 0: return 1.0
    from math import comb
    pk = [comb(n, i) * 0.5 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in pk if p <= pk[k] + 1e-15))

real = [r for r in rows if r['set'] != 'random']; rnd = [r for r in rows if r['set'] == 'random']
out = {}
max_diff = max(float(r['max_abs_diff']) for r in ec)
out['R1'] = dict(pass_=all(r['match'] == '1' for r in bf) and max_diff <= 1e-6, brute=[dict(r) for r in bf],
                 max_energy_abs_diff=max_diff, n_over_tol=sum(float(r['max_abs_diff']) > 1e-6 for r in ec),
                 n_instances=len(ec), brute_part_pass=all(r['match'] == '1' for r in bf))
gaps = [r['eng_gap8'] for r in rows]
fin = [g for g in gaps if not math.isinf(g)]
out['R2'] = dict(pass_=all(r['eng_hit8'] for r in real) and np.mean([r['eng_hit8'] for r in rnd]) >= 0.95 and
                 (np.mean(gaps) if not any(math.isinf(g) for g in gaps) else math.inf) <= 0.01,
                 real_hit8={r['id']: bool(r['eng_hit8']) for r in real}, random_hit8_frac=float(np.mean([r['eng_hit8'] for r in rnd])),
                 mean_gap8_all=(float(np.mean(gaps)) if not fin or len(fin) == len(gaps) else 'inf'),
                 n_best8_infeasible=sum(math.isinf(g) for g in gaps), mean_gap8_over_feasible=float(np.mean(fin)) if fin else None,
                 neal_real_hit8={r['id']: bool(r['neal_hit8']) for r in real}, neal_random_hit8_frac=float(np.mean([r['neal_hit8'] for r in rnd])),
                 neal_n_best8_infeasible=sum(math.isinf(r['neal_gap8']) for r in rows))
W = sum(r['r3'] == 'win' for r in rows); L = sum(r['r3'] == 'loss' for r in rows); T = sum(r['r3'] == 'tie' for r in rows)
p = binom_two_sided(min(W, L), W + L)
verdict = 'tie' if p >= 0.05 else ('engine beats neal' if W > L else 'neal beats engine')
out['R3'] = dict(wins=W, ties=T, losses=L, sign_test_p=p, verdict=verdict)
both = [r for r in rows if r['eng_ps'] > 0 and r['neal_ps'] > 0]
ratio = [r['eng_tts99_s'] / r['neal_tts99_s'] for r in both]
med = float(np.median(ratio)) if ratio else None
out['R4'] = dict(n_both_ps_pos=len(both), median_tts_ratio_engine_over_neal=med,
                 verdict=(None if med is None else ('engine faster' if med <= 0.5 else ('engine slower' if med >= 2 else 'comparable'))),
                 side_tally_engine_only_ps_pos=sum(r['eng_ps'] > 0 and r['neal_ps'] == 0 for r in rows),
                 side_tally_neal_only_ps_pos=sum(r['neal_ps'] > 0 and r['eng_ps'] == 0 for r in rows),
                 n_neither=sum(r['neal_ps'] == 0 and r['eng_ps'] == 0 for r in rows))
gs = [r for r in rows if r['greedy_gap'] > 1e-9]
out['R5'] = dict(pass_=all(r['eng_hit8'] for r in gs), n_greedy_suboptimal=len(gs), n_instances=len(rows),
                 engine_hit8_on_those=sum(bool(r['eng_hit8']) for r in gs), neal_hit8_on_those=sum(bool(r['neal_hit8']) for r in gs),
                 greedy_mean_gap=float(np.mean([r['greedy_gap'] for r in rows])), greedy_max_gap=float(max(r['greedy_gap'] for r in rows)))
per_n = {}
for n in sorted({r['n'] for r in rows}):
    rr = [r for r in rows if r['n'] == n]
    per_n[n] = dict(k=len(rr), nvars=rr[0]['nvars'], eng_hit8=sum(bool(r['eng_hit8']) for r in rr), neal_hit8=sum(bool(r['neal_hit8']) for r in rr),
                    eng_ps_mean=float(np.mean([r['eng_ps'] for r in rr])), neal_ps_mean=float(np.mean([r['neal_ps'] for r in rr])),
                    eng_feas_mean=float(np.mean([r['eng_feas'] for r in rr])), neal_feas_mean=float(np.mean([r['neal_feas'] for r in rr])),
                    eng_t_us=float(np.mean([r['eng_t_us'] for r in rr])), neal_t_us=float(np.mean([r['neal_t_us'] for r in rr])),
                    greedy_subopt=sum(r['greedy_gap'] > 1e-9 for r in rr),
                    eng_gap8_bestfeas_med=float(np.median([r['eng_gap8_bestfeas'] for r in rr])),
                    neal_gap8_bestfeas_med=float(np.median([r['neal_gap8_bestfeas'] for r in rr])))
out['per_n'] = per_n
out['real'] = real
json.dump(out, open(os.path.join(d, 'ree_rules.json'), 'w'), indent=1, default=str)
for k in ['R1', 'R2', 'R3', 'R4', 'R5']:
    print(k, {kk: vv for kk, vv in out[k].items() if kk not in ('brute',)})
print('per n:')
for n, v in per_n.items(): print(n, v)
for r in real: print(r)
