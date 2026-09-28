"""ADR-012 part A statistics: frozen rules A1-A5 (+ exploratory whitened statistic).
Usage: python tools/tunnel_stats.py <dir with tunnel_trials.csv> [--exploratory]
Writes tunnel_cells.csv and tunnel_rules.json into the same directory."""
import csv, json, sys, math
import numpy as np

d = sys.argv[1]
expl = '--exploratory' in sys.argv
rows = list(csv.DictReader(open(f"{d}/tunnel_trials.csv")))
ph = list(csv.DictReader(open(f"{d}/tunnel_photons.csv")))
Vs = ['inf', '50', '30', '20', '15', '10', '5']
Bs = sorted({r['B'] for r in rows}, key=float)
key = lambda r: (r['sensor'], r['V'], r['B'], r['pipe'])
by = {}
for r in rows:
    by.setdefault(key(r) + (r['scene'],), []).append(r)
netph = {}
for r in ph:
    netph.setdefault((r['sensor'], r['V'], r['B'], r['scene']), []).append(float(r['net_vs_empty']))

def zstats(t, e):
    t, e = np.asarray(t), np.asarray(e)
    sd = e.std(ddof=1)
    z = (np.median(t) - e.mean()) / sd if sd > 0 else float('nan')
    auc = np.mean([(1.0 if a > b else 0.5 if a == b else 0.0) for a in t for b in e])
    return float(z), float(auc)

cells = []
for sn in ['far', 'near']:
    for V in Vs:
        for B in Bs:
            for pipe in ['P1', 'P2']:
                emp = by[(sn, V, B, pipe, 'empty')]
                for tg in ['person', 'vehicle']:
                    tr = by[(sn, V, B, pipe, tg)]
                    z, auc = zstats([float(r['T']) for r in tr], [float(r['T']) for r in emp])
                    zw, aucw = zstats([float(r['Tw']) for r in tr], [float(r['Tw']) for r in emp])
                    dist = np.array([float(r['dist']) for r in tr]); wd = np.array([float(r['wdist']) for r in tr])
                    c = dict(sensor=sn, V=V, B=B, pipe=pipe, target=tg, n=len(tr), Z=z, AUC=auc,
                             loc_hits=int((dist <= 0.5).sum()), med_dist=float(np.median(dist)),
                             Zw=zw, AUCw=aucw, locw_hits=int((wd <= 0.5).sum()), med_wdist=float(np.median(wd)),
                             net_photons_med=float(np.median(netph[(sn, V, B, tg)])))
                    c['detected'] = bool(c['Z'] >= 5 and c['AUC'] >= 0.99)
                    c['localised'] = c['loc_hits'] >= 7
                    c['detected_w'] = bool(c['Zw'] >= 5 and c['AUCw'] >= 0.99)
                    c['localised_w'] = c['locw_hits'] >= 7
                    cells.append(c)
with open(f"{d}/tunnel_cells.csv", 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(cells[0].keys())); w.writeheader(); w.writerows(cells)

def cell(sn, V, B, pipe, tg):
    return next(c for c in cells if (c['sensor'], c['V'], c['B'], c['pipe'], c['target']) == (sn, V, B, pipe, tg))

def vbreak(sn, B, pipe, tg, sfx=''):
    vb = 'none'
    for V in Vs:
        c = cell(sn, V, B, pipe, tg)
        if c['detected' + sfx] and c['localised' + sfx]:
            vb = V
        else:
            break
    return vb

out = {'dir': d, 'budgets': Bs, 'exploratory_run': expl}
B12 = next((b for b in Bs if float(b) == 1e12), None)
vb_all = {}
for sfx, tag in [('', 'frozen_T'), ('_w', 'exploratory_whitened_T')]:
    res = {}
    vb_all[tag] = {f"{tg}|{sn}|{pipe}|B={B}": vbreak(sn, B, pipe, tg, sfx)
                   for tg in ['person', 'vehicle'] for sn in ['far', 'near'] for pipe in ['P1', 'P2'] for B in Bs}
    if B12 is not None:
        cp, cv = cell('far', 'inf', B12, 'P2', 'person'), cell('far', 'inf', B12, 'P2', 'vehicle')
        a1 = all(c['detected' + sfx] and c['localised' + sfx] for c in (cp, cv))
        res['A1'] = dict(pass_=a1, person=cp, vehicle=cv)
        p1p, p1v = cell('far', 'inf', B12, 'P1', 'person'), cell('far', 'inf', B12, 'P1', 'vehicle')
        res['A2'] = dict(person_usable=bool(p1p['detected' + sfx] and p1p['localised' + sfx]),
                         vehicle_usable=bool(p1v['detected' + sfx] and p1v['localised' + sfx]), person=p1p, vehicle=p1v)
        vbf = vbreak('far', B12, 'P2', 'person', sfx); vbn = vbreak('near', B12, 'P2', 'person', sfx)
        res['A3'] = dict(pass_=vbf in ('15', '20', '30'), V_break_person_far_P2_B1e12=vbf)
        order = {v: i for i, v in enumerate(Vs)}
        a5 = (vbn != 'none') and (vbf == 'none' or order[vbn] > order[vbf])
        res['A5'] = dict(pass_=bool(a5), V_break_far=vbf, V_break_near=vbn)
    det5 = [c for c in cells if c['V'] == '5' and c['detected' + sfx]]
    res['A4'] = dict(pass_=len(det5) == 0, detected_at_V5=[(c['sensor'], c['B'], c['pipe'], c['target'], c['Z']) for c in det5])
    out[tag] = res
out['V_break'] = vb_all
json.dump(out, open(f"{d}/tunnel_rules.json", 'w'), indent=1, default=str)
# console summary
for tag in ['frozen_T', 'exploratory_whitened_T']:
    r = out[tag]
    print(f"== {tag}")
    for k in ['A1', 'A2', 'A3', 'A4', 'A5']:
        if k in r:
            print(k, {kk: vv for kk, vv in r[k].items() if kk not in ('person', 'vehicle')})
print("cells detected&localised (frozen):", [(c['sensor'], c['V'], c['B'], c['pipe'], c['target'], round(c['Z'], 1)) for c in cells if c['detected'] and c['localised']])
print("cells detected&localised (whitened):", [(c['sensor'], c['V'], c['B'], c['pipe'], c['target'], round(c['Zw'], 1)) for c in cells if c['detected_w'] and c['localised_w']])
