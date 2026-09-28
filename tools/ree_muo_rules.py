"""ADR-014 decision rules from check_ree_muo outputs (frozen rules, section 6).
Usage: python tools/ree_muo_rules.py <experiment dir>"""
import csv, json, math, sys, os

D = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(D, "summary_detection.csv"))))
loc = list(csv.DictReader(open(os.path.join(D, "localisation.csv"))))
kv = {}
for line in open(os.path.join(D, "flux_sanity.txt")):
    p = line.split()
    if len(p) >= 2:
        kv[p[0]] = p[1:]
val = open(os.path.join(D, "validation.txt")).read()
blind = {l.split()[0]: l.split()[1:] for l in open(os.path.join(D, "blind_truth.txt")) if l.strip()}
best = {l.split()[0]: l.split()[1:] for l in open(os.path.join(D, "blind_estimate.txt")) if l.strip()}

cells = {}
for r in rows:
    c = int(r["cell"])
    e = cells.setdefault(c, dict(d=float(r["d"]), L=float(r["L"]), drho=float(r["drho"]), T={}))
    e["T"][float(r["T_days"])] = dict(Z=float(r["Z_emp"]), AUC=float(r["AUC"]), Zi=float(r["Z_asimov_ideal"]),
                                      Zn=float(r["Z_asimov_nuis"]), det=int(r["detected"]))
for c, e in cells.items():
    e["det_any"] = any(v["det"] for T, v in e["T"].items() if T <= 180)
    e["det180"] = e["T"][180.0]["det"]
    e["first_T"] = min([T for T, v in e["T"].items() if v["det"]], default=None)

def cell(d, L, dr):
    for c, e in cells.items():
        if e["d"] == d and e["L"] == L and abs(e["drho"] - dr) < 1e-6:
            return c, e
    raise KeyError((d, L, dr))

depths = [40.0, 80.0, 120.0, 160.0]; Ls = [5.0, 10.0, 20.0, 40.0]; drs = [0.05, 0.25, 0.60]
out = {}
# R1
fv = "FROZEN" in val and " PASS" in val.splitlines()[0]
ratio = float(kv["ratio_sim_over_MH1"][0])
out["R1"] = dict(bernoulli_line=val.splitlines()[0], bernoulli_pass=fv, sim_vertical_540=float(kv["sim_vertical_540_cm2_s_sr"][0]),
                 MH_eq1=float(kv["MeiHime_eq1_vertical_0p54"][0]), ratio=ratio, flux_pass=0.5 <= ratio <= 2.0,
                 total_ratio_eq4=float(kv["ratio_total_sim_over_MH4"][0]), pass_=fv and 0.5 <= ratio <= 2.0,
                 counts_per_day=kv.get("K0_mean_counts_per_day_in_acceptance"))
# R2
r2 = {int(d): cell(d, 20.0, 0.60)[1]["det_any"] for d in depths}
out["R2"] = dict(per_depth=r2, pass_=all(r2.values()))
# R3
Lmin = {}
for dr in drs:
    for d in depths:
        lm = None
        for L in reversed(Ls):
            if cell(d, L, dr)[1]["det_any"]:
                lm = L
            else:
                break
        Lmin[f"{dr:.2f}_{int(d)}"] = lm
def le(x, v):
    return x is not None and x <= v
s1 = all(le(Lmin[f"0.60_{d}"], 10) for d in (120, 160))
s2 = all(le(Lmin[f"0.60_{d}"], 20) for d in (40, 80, 120, 160))
s3 = all(le(Lmin[f"0.25_{d}"], 20) for d in (120, 160))
s4 = all(le(Lmin[f"0.25_{d}"], 40) for d in (40, 80, 120, 160))
out["R3"] = dict(L_min=Lmin, stmt_ore_le10_deep=s1, stmt_ore_le20_all=s2, stmt_carb_le20_deep=s3, stmt_carb_le40_all=s4,
                 pass_=s1 and s2 and s3 and s4)
# R4
r4 = {f"{int(d)}_{int(L)}": cell(d, L, 0.05)[1]["det_any"] for d in depths for L in (5.0, 10.0, 20.0)}
r4_40 = {int(d): cell(d, 40.0, 0.05)[1]["det_any"] for d in depths}
out["R4"] = dict(detected_L_le20=r4, detected_L40=r4_40, pass_=not any(r4.values()))
# R5
nok = {}
errs = {}
for r in loc:
    c = int(r["cell"])
    nok[c] = nok.get(c, 0) + int(r["within"])
    errs.setdefault(c, []).append(float(r["dist_m"]))
def med(v):
    v = sorted(v); n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])
r5 = {}
for c, e in sorted(cells.items()):
    if e["det180"]:
        r5[c] = dict(d=e["d"], L=e["L"], drho=e["drho"], within=nok[c], median_err=round(med(errs[c]), 2),
                     ok=nok[c] >= 7)
out["R5"] = dict(detected_cells=r5, n_detected=len(r5), n_ok=sum(v["ok"] for v in r5.values()),
                 pass_=len(r5) > 0 and all(v["ok"] for v in r5.values()))
out["localisation_all"] = {c: dict(d=cells[c]["d"], L=cells[c]["L"], drho=cells[c]["drho"], within=nok[c],
                                    errs=[round(x, 2) for x in errs[c]]) for c in sorted(cells)}
# R6
out["R6"] = dict(truth=blind["truth"], estimate=best["estimate"], error_m=float(blind["error_m"][0]),
                 pass_=float(blind["error_m"][0]) <= 10.0)
# R7
out["R7"] = {f"d{int(d)}_L{int(L)}": dict(Z45=cell(d, L, 0.60)[1]["T"][45.0]["Z"], AUC45=cell(d, L, 0.60)[1]["T"][45.0]["AUC"],
                                          Zideal45=cell(d, L, 0.60)[1]["T"][45.0]["Zi"]) for d in depths for L in Ls}
out["cells"] = {c: dict(d=e["d"], L=e["L"], drho=e["drho"], det_any=e["det_any"], det180=e["det180"], first_T=e["first_T"],
                        Z={int(T): round(v["Z"], 2) for T, v in sorted(e["T"].items())},
                        AUC={int(T): round(v["AUC"], 3) for T, v in sorted(e["T"].items())},
                        Zideal={int(T): round(v["Zi"], 2) for T, v in sorted(e["T"].items())},
                        Znuis={int(T): round(v["Zn"], 2) for T, v in sorted(e["T"].items())}) for c, e in sorted(cells.items())}
json.dump(out, open(os.path.join(D, "ree_muo_rules.json"), "w"), indent=1, default=str)
for k in ("R1", "R2", "R3", "R4", "R5", "R6"):
    v = out[k]
    print(k, "PASS" if v["pass_"] else "FAIL", {kk: vv for kk, vv in v.items() if kk not in ("pass_", "detected_cells")})
print("R5 detected cells:", {c: (v["d"], v["L"], v["drho"], v["within"]) for c, v in out["R5"]["detected_cells"].items()})
print("R7:", {k: round(v["Z45"], 2) for k, v in out["R7"].items()})
