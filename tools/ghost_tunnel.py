"""ADR-012 part B: computational ghost imaging (Walsh-Hadamard, bucket) vs camera through dust.
Analytic linear image-formation model exactly as frozen in ADR-012 sec. 7.
Usage: python tools/ghost_tunnel.py <outdir> [seeds=8] [--return-only]
--return-only is EXPLORATORY (not frozen): dust only between target and detector (Tajahuerce-like
geometry): clean illumination for both systems, no backscatter veil."""
import sys, os, json, csv, time
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else 'build/ghost'
nseeds = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 8
RETURN_ONLY = '--return-only' in sys.argv
os.makedirs(out, exist_ok=True)
n = 64; NP = n * n
TAUS = [0, 0.5, 1, 2, 3, 4, 5, 6]
BUDGETS = [1e5, 1e7, 1e9]
OMEGA0, ETA_F, S0, V0, READ = 0.9, 0.5, 1.5, 0.05, 2.0
KS = [1e-4, 1e-3, 1e-2, 1e-1]

def make_target():
    R = np.full((n, n), 0.05)
    yy, xx = np.mgrid[0:n, 0:n]
    # person silhouette (left half)
    R[((xx - 16) ** 2 + (yy - 12) ** 2) <= 16] = 0.9            # head r=4
    R[17:38, 10:23] = 0.9                                        # torso
    R[38:58, 11:15] = 0.9; R[38:58, 18:22] = 0.9                 # legs
    R[18:34, 6:9] = 0.9; R[18:34, 24:27] = 0.9                   # arms
    # bar chart (right half): bar pairs of width 1,2,3,4 px
    x = 34
    for w in [1, 2, 3, 4]:
        for _ in range(2):
            R[8:30, x:x + w] = 0.6; x += 2 * w
        x += 2
    # a grid of dots (bottom right)
    for cy in range(38, 60, 6):
        for cx in range(36, 62, 6):
            R[cy:cy + 2, cx:cx + 2] = 0.6
    return R

def gauss_kernel(s):
    r = max(1, int(np.ceil(4 * s)))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / s) ** 2); return k / k.sum(), r

def blur(img, s, mode='edge'):
    """Separable Gaussian blur on the last two axes, 'nearest' (edge) boundary -> flat stays flat."""
    if s <= 0: return img.copy()
    k, r = gauss_kernel(s)
    a = np.pad(img, [(0, 0)] * (img.ndim - 2) + [(r, r), (0, 0)], mode=mode)
    a = sum(k[i] * a[..., i:i + img.shape[-2], :] for i in range(len(k)))
    a = np.pad(a, [(0, 0)] * (img.ndim - 2) + [(0, 0), (r, r)], mode=mode)
    return sum(k[i] * a[..., :, i:i + img.shape[-1]] for i in range(len(k)))

def transfer(I, tau):
    """T[I] = e^-tau I + (1 - e^-tau) omega0 eta_f (G_s * I), s = 1.5 sqrt(tau)."""
    if tau == 0: return I.copy()
    return np.exp(-tau) * I + (1 - np.exp(-tau)) * OMEGA0 * ETA_F * blur(I, S0 * np.sqrt(tau))

def fwht(a):
    """Unnormalised fast Walsh-Hadamard transform along the last axis (Sylvester order)."""
    a = a.copy(); h = 1; N = a.shape[-1]
    while h < N:
        a = a.reshape(a.shape[:-1] + (N // (2 * h), 2, h))
        x, y = a[..., 0, :].copy(), a[..., 1, :].copy()
        a[..., 0, :], a[..., 1, :] = x + y, x - y
        a = a.reshape(a.shape[:-3] + (N,)); h *= 2
    return a

def affine_fit(img, truth):
    A = np.stack([img.ravel(), np.ones(img.size)], 1)
    c, *_ = np.linalg.lstsq(A, truth.ravel(), rcond=None)
    return (A @ c).reshape(truth.shape)

def psnr(img, truth):
    return 10 * np.log10(1.0 / np.mean((img - truth) ** 2))

def ssim(x, y):
    C1, C2 = (0.01) ** 2, (0.03) ** 2
    f = lambda z: blur(z, 1.5, mode='reflect')
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx * mx, f(y * y) - my * my, f(x * y) - mx * my
    m = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))
    return float(m.mean())

def otf(tau):
    """OTF of the one-way transfer (periodic approximation, used only for Wiener)."""
    fy = np.fft.fftfreq(n)[:, None]; fx = np.fft.fftfreq(n)[None, :]
    if tau == 0: return np.ones((n, n))
    s = S0 * np.sqrt(tau)
    G = np.exp(-2 * np.pi ** 2 * s ** 2 * (fx ** 2 + fy ** 2))
    return np.exp(-tau) + (1 - np.exp(-tau)) * OMEGA0 * ETA_F * G

def wiener_best(img, tau, truth):
    H = otf(tau); Y = np.fft.fft2(img - img.mean())
    best = None
    for K in KS:
        X = np.real(np.fft.ifft2(np.conj(H) * Y / (np.abs(H) ** 2 + K))) + img.mean()
        fit = affine_fit(X, truth); p = psnr(fit, truth)
        if best is None or p > best[0]: best = (p, fit, K)
    return best

t_start = time.time()
R = make_target()
# Sylvester Hadamard rows via FWHT of the identity (h_k = row k)
Hm = fwht(np.eye(NP, dtype=np.float64))                 # symmetric, entries +-1
Hm = Hm.astype(np.float32)
rows, imgs = [], {}
for tau in TAUS:
    # ---- noiseless physics (per unit budget) ----
    eta_ret = np.exp(-tau) + (1 - np.exp(-tau)) * OMEGA0 * ETA_F
    veil_frac = 0.0 if RETURN_ONLY else V0 * (1 - np.exp(-2 * tau))
    tau_ill = 0.0 if RETURN_ONLY else tau
    flat_T = transfer(np.ones((n, n)), tau_ill)
    cam_unit = transfer(R * flat_T, tau) / NP               # x N -> expected counts per pixel
    cam_veil_unit = veil_frac / NP
    # GI: each pair illuminates every pixel once with a = N / NP^2
    sig_p = np.empty(NP); sig_m = np.empty(NP); onp = np.empty(NP)
    for c0 in range(0, NP, 512):
        h = Hm[c0:c0 + 512].reshape(-1, n, n).astype(np.float64)
        mp, mm = 0.5 * (1 + h), 0.5 * (1 - h)
        sig_p[c0:c0 + 512] = (R[None] * transfer(mp, tau_ill)).sum((1, 2))
        sig_m[c0:c0 + 512] = (R[None] * transfer(mm, tau_ill)).sum((1, 2))
        onp[c0:c0 + 512] = mp.sum((1, 2))
    onm = NP - onp
    gi_p_unit = (eta_ret * sig_p + veil_frac * onp) / NP ** 2   # x N -> expected bucket counts
    gi_m_unit = (eta_ret * sig_m + veil_frac * onm) / NP ** 2
    for bi, N in enumerate(BUDGETS):
        for seed in range(1, nseeds + 1):
            rng = np.random.default_rng([12, seed, bi, int(tau * 10)])
            cam = rng.poisson(N * (cam_unit + cam_veil_unit)).astype(float) + rng.normal(0, READ, (n, n))
            bp = rng.poisson(N * gi_p_unit).astype(float) + rng.normal(0, READ, NP)
            bm = rng.poisson(N * gi_m_unit).astype(float) + rng.normal(0, READ, NP)
            gi = fwht(bp - bm).reshape(n, n)                      # x = H (b+ - b-)
            res = {}
            for name, im in [('camera', cam), ('ghost', gi)]:
                fit = affine_fit(im, R)
                res[name] = (psnr(fit, R), ssim(fit, R), fit)
                pw, fw, K = wiener_best(im, (0.0 if (RETURN_ONLY and name == 'ghost') else tau), R)
                res[name + '_wiener'] = (pw, ssim(fw, R), fw)
            for name, (p, s, fit) in res.items():
                rows.append(dict(seed=seed, N=N, tau=tau, method=name, psnr=p, ssim=s))
                if seed == 1:
                    imgs[f"{name}|{N:g}|{tau}"] = fit.astype(np.float32)
    print(f"tau {tau} done ({time.time() - t_start:.1f} s)", flush=True)

with open(f"{out}/ghost_results.csv", 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
np.savez_compressed(f"{out}/ghost_images_seed1.npz", truth=R.astype(np.float32), **imgs)

# ---- rules B1, B1-pred, B2 ----
def beats(N, tau, a, b):
    pa = {r['seed']: r['psnr'] for r in rows if r['N'] == N and r['tau'] == tau and r['method'] == a}
    pb = {r['seed']: r['psnr'] for r in rows if r['N'] == N and r['tau'] == tau and r['method'] == b}
    wins = sum(pa[s] >= pb[s] + 1.0 for s in pa)
    return bool(wins >= 7), int(wins), float(np.median([pa[s] - pb[s] for s in pa]))

def claim(N, a, b):
    per = {tau: beats(N, tau, a, b) for tau in TAUS}
    X = None
    for i, tau in enumerate(TAUS):
        if all(per[t][0] for t in TAUS[i:]):
            X = tau; break
    return per, X

rules = {}
per, X = claim(1e7, 'ghost', 'camera')
rules['B1'] = dict(X=X, claim=(f"GI beats camera at OD >= {X}" if X is not None else
                               "no OD in [0, 6] where GI beats the camera by >= 1 dB"),
                   per_tau={str(t): dict(beats=v[0], wins_of_8=v[1], median_dPSNR_dB=v[2]) for t, v in per.items()})
rules['B1_pred'] = dict(pass_=not any(per[t][0] for t in TAUS if t <= 2),
                        statement="GI does NOT beat the camera at any tau <= 2 at N = 1e7")
rules['B2'] = {}
for N in [1e5, 1e9]:
    p2, X2 = claim(N, 'ghost', 'camera')
    rules['B2'][f"N={N:g}"] = dict(X=X2, per_tau={str(t): dict(beats=v[0], wins_of_8=v[1], median_dPSNR_dB=v[2]) for t, v in p2.items()})
p3, X3 = claim(1e7, 'ghost_wiener', 'camera_wiener')
rules['B2']['wiener_N=1e7'] = dict(X=X3, per_tau={str(t): dict(beats=v[0], wins_of_8=v[1], median_dPSNR_dB=v[2]) for t, v in p3.items()})
# reverse direction for context: camera beats GI by >= 1 dB
p4, X4 = claim(1e7, 'camera', 'ghost')
rules['context_camera_beats_ghost_N=1e7'] = {str(t): dict(beats=v[0], wins_of_8=v[1], median_dPSNR_dB=v[2]) for t, v in p4.items()}
rules['runtime_s'] = time.time() - t_start
rules['return_only_exploratory'] = RETURN_ONLY
json.dump(rules, open(f"{out}/ghost_rules.json", 'w'), indent=1)
print(json.dumps({k: (v if k != 'B2' else {kk: vv['X'] for kk, vv in v.items()}) for k, v in rules.items() if k in ('B1_pred', 'B2')}, indent=1))
print('B1:', rules['B1']['claim'])
for t, v in rules['B1']['per_tau'].items(): print('  tau', t, v)
