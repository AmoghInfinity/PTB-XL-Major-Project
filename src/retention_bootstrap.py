"""Patient-cluster bootstrap for tier retention rho_tier(c), Hamming risk and macro-F1 at fixed coverage.

Usage: python retention_bootstrap.py <results_dir> <tiers.json> <ptbxl_database.csv> <out.json>
"""
import sys, json
import numpy as np
import pandas as pd

R, TJ, MD, OUT = sys.argv[1:5]
T = json.load(open(TJ)); tier = np.array([T["tier_of"][v] for v in T["vocab"]])
md = pd.read_csv(MD, index_col="ecg_id"); pid = md.loc[md["strat_fold"] == 10, "patient_id"].to_numpy()
FILES = {"ResNet1D": "baseline_test_predictions.npz", "ECG-Mamba": "ecg_mamba_test_predictions.npz"}
COVS = (0.8, 0.5); TIERS = ("common", "medium", "rare")


def stats(y, p):
    n = len(y); q = np.clip(p, 1e-12, 1 - 1e-12)
    H = (-(q * np.log(q) + (1 - q) * np.log(1 - q))).mean(1)
    order = np.argsort(H, kind="stable"); pred = p >= 0.5
    out = {}
    for c in COVS:
        sel = order[: int(np.ceil(c * n))]
        for t in TIERS:
            m = tier == t; tot = y[:, m].sum()
            out[f"rho_{t}@{c}"] = y[sel][:, m].sum() / tot if tot else np.nan
        ys, ps = y[sel], pred[sel]
        tp = (ys & ps).sum(0); fp = (~ys & ps).sum(0); fn = (ys & ~ps).sum(0)
        den = 2 * tp + fp + fn
        out[f"macro_f1@{c}"] = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0).mean()
        out[f"risk@{c}"] = (ps != ys).mean()
    return out


D = {}
for m, f in FILES.items():
    d = np.load(f"{R}/{f}"); D[m] = (d["y_true"].astype(bool), d["y_prob"])
assert len(pid) == len(D["ResNet1D"][0])
obs = {m: stats(*D[m]) for m in D}
groups = [np.flatnonzero(pid == u) for u in np.unique(pid)]
rng = np.random.default_rng(42); boot = {m: [] for m in D}
for _ in range(1000):
    idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
    for m in D:
        boot[m].append(stats(D[m][0][idx], D[m][1][idx]))
res = {"resampling_unit": "patient", "n_boot": 1000, "interval": "percentile 95%", "models": {},
       "diff_ECG-Mamba_minus_ResNet1D": {}}
for m in D:
    res["models"][m] = {}
    for k, v in obs[m].items():
        b = np.array([x[k] for x in boot[m]], float); b = b[np.isfinite(b)]
        res["models"][m][k] = {"estimate": float(v), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
for k in obs["ResNet1D"]:
    d = np.array([a[k] - b[k] for a, b in zip(boot["ECG-Mamba"], boot["ResNet1D"])], float); d = d[np.isfinite(d)]
    res["diff_ECG-Mamba_minus_ResNet1D"][k] = {"estimate": float(obs["ECG-Mamba"][k] - obs["ResNet1D"][k]),
                                              "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
json.dump(res, open(OUT, "w"), indent=1)
for m in D:
    print(m, " | ".join(f"{k}={v['estimate']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}]" for k, v in res["models"][m].items()))
print("diff", " | ".join(f"{k}={v['estimate']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]"
                         for k, v in res["diff_ECG-Mamba_minus_ResNet1D"].items()))
