"""
Post-hoc analyses for the manuscript, computed ONLY from saved prediction files.

Usage (from the project root):
    python src/paper_analysis.py --results results --out results/paper_analysis
    # optional patient-cluster bootstrap CIs (needs PTB-XL metadata):
    python src/paper_analysis.py --results results --out results/paper_analysis \
        --metadata data/ptbxl/ptbxl_database.csv --tiers-from-checkpoint checkpoints/best_model.pt

Without --metadata, only point estimates are produced.
"""
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

TIERS = ("common", "medium", "rare")
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
EPS = 1e-12

MODELS = {
    # name: (test raw, test temperature-scaled)
    "ResNet1D": ("baseline_test_predictions.npz", "temperature_scaled_test_predictions.npz"),
    "ECG-Mamba": ("ecg_mamba_test_predictions.npz", "ecg_mamba_temperature_scaled_test_predictions.npz"),
}


def load(path):
    d = np.load(path, allow_pickle=False)
    return d["y_true"].astype(np.int64), d["y_prob"].astype(np.float64), d["vocab"].astype(str)


def ece_equal_width(y, p, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    out = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return out


def ece_equal_mass(y, p, n_bins=15):
    order = np.argsort(p, kind="stable")
    out = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            out += len(chunk) / len(p) * abs(y[chunk].mean() - p[chunk].mean())
    return out


def per_label(y, p):
    L = y.shape[1]
    r = {k: np.full(L, np.nan) for k in ("auroc", "auprc", "f1", "ece", "aece", "brier", "prev", "oe", "ece_rel")}
    for j in range(L):
        yj, pj = y[:, j], p[:, j]
        npos = yj.sum()
        if 0 < npos < len(yj):
            r["auroc"][j] = roc_auc_score(yj, pj)
            r["auprc"][j] = average_precision_score(yj, pj)
        r["f1"][j] = f1_score(yj, pj >= 0.5, zero_division=0)
        r["ece"][j] = ece_equal_width(yj, pj)
        r["aece"][j] = ece_equal_mass(yj, pj)
        r["brier"][j] = np.mean((pj - yj) ** 2)
        r["prev"][j] = yj.mean()
        if npos > 0:
            r["oe"][j] = pj.sum() / npos            # expected / observed positives
            r["ece_rel"][j] = r["ece"][j] / yj.mean()  # ECE in units of prevalence
    return r


def tier_summary(r, tier):
    out = {}
    for t in TIERS:
        m = tier == t
        for k in ("auroc", "auprc", "f1", "ece", "aece", "brier", "ece_rel"):
            out[f"{k}_{t}"] = float(np.nanmean(r[k][m]))
        # geometric mean of E/O ratio (ratio quantity)
        oe = r["oe"][m]
        oe = oe[np.isfinite(oe) & (oe > 0)]
        out[f"oe_gmean_{t}"] = float(np.exp(np.mean(np.log(oe)))) if len(oe) else float("nan")
    for k in ("auroc", "auprc", "f1", "ece", "aece", "brier", "ece_rel"):
        out[f"{k}_macro"] = float(np.nanmean(r[k]))
    return out


def entropy_score(p):
    q = np.clip(p, EPS, 1 - EPS)
    return (-(q * np.log(q) + (1 - q) * np.log(1 - q)) / np.log(2)).mean(1)


def selective(y, p, tier, score=None):
    n = len(y)
    pred = (p >= 0.5).astype(np.int64)
    risk = (pred != y).mean(1)
    s = entropy_score(p) if score is None else score
    order = np.argsort(s, kind="stable")
    cum = np.cumsum(risk[order]) / np.arange(1, n + 1)
    cov = np.arange(1, n + 1) / n
    fc, fr = np.r_[0, cov], np.r_[0, cum]
    aurc = float(np.sum(np.diff(fc) * (fr[:-1] + fr[1:]) / 2))
    o_or = np.argsort(risk, kind="stable")
    cum_or = np.cumsum(risk[o_or]) / np.arange(1, n + 1)
    fr_or = np.r_[0, cum_or]
    aurc_or = float(np.sum(np.diff(fc) * (fr_or[:-1] + fr_or[1:]) / 2))
    out = {"aurc": aurc, "aurc_oracle": aurc_or, "e_aurc": aurc - aurc_or, "full_risk": float(risk.mean())}
    pos_tot = {t: y[:, tier == t].sum() for t in TIERS}
    curve = {"coverage": cov.tolist(), "risk": cum.tolist()}
    for c in COVERAGES:
        k = int(np.ceil(c * n)); sel = order[:k]
        key = f"{c:.1f}"
        out[f"risk@{key}"] = float(cum[k - 1])
        out[f"macro_f1@{key}"] = float(f1_score(y[sel], pred[sel], average="macro", zero_division=0))
        out[f"micro_f1@{key}"] = float(f1_score(y[sel], pred[sel], average="micro", zero_division=0))
        for t in TIERS:
            out[f"retained_pos_{t}@{key}"] = float(y[sel][:, tier == t].sum() / pos_tot[t])
        out[f"retained_records_with_any_label@{key}"] = float((y[sel].sum(1) > 0).mean())
    # dense retention curves for plotting
    ret = {}
    for t in TIERS:
        cs = np.cumsum(y[order][:, tier == t].sum(1))
        ret[t] = (cs / pos_tot[t]).tolist()
    curve["retained"] = ret
    return out, curve


def scalars(y, praw, pts, tier):
    out = {}
    rr, rt = per_label(y, praw), per_label(y, pts)
    for k, v in tier_summary(rr, tier).items():
        out[f"raw_{k}"] = v
    for k, v in tier_summary(rt, tier).items():
        out[f"ts_{k}"] = v
    s_raw, _ = selective(y, praw, tier)
    s_ts, _ = selective(y, pts, tier)
    for k, v in s_raw.items():
        out[f"sel_raw_{k}"] = v
    out["sel_ts_aurc"] = s_ts["aurc"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiers-json", help="JSON with {'vocab':[...], 'tier_of':{...}}")
    ap.add_argument("--tiers-from-checkpoint", help="checkpoint .pt containing vocab and tier_of (needs torch)")
    ap.add_argument("--metadata", help="ptbxl_database.csv for patient-cluster bootstrap")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    R, O = Path(a.results), Path(a.out); O.mkdir(parents=True, exist_ok=True)

    if a.tiers_json:
        T = json.load(open(a.tiers_json))
    else:
        import torch
        c = torch.load(a.tiers_from_checkpoint, map_location="cpu", weights_only=False)
        T = {"vocab": list(c["vocab"]), "tier_of": dict(c["tier_of"])}
    vocab = T["vocab"]; tier = np.array([T["tier_of"][v] for v in vocab])

    results, curves, label_table = {}, {}, {}
    data = {}
    for name, (fr, ft) in MODELS.items():
        y, praw, v1 = load(R / fr); y2, pts, v2 = load(R / ft)
        assert list(v1) == vocab == list(v2) and np.array_equal(y, y2)
        data[name] = (y, praw, pts)
        results[name] = scalars(y, praw, pts, tier)
        _, curves[name] = selective(y, praw, tier)
        r = per_label(y, praw)
        label_table[name] = {k: [None if not np.isfinite(x) else float(x) for x in r[k]] for k in r}
    y = data["ResNet1D"][0]
    label_table["test_positives"] = y.sum(0).astype(int).tolist()
    label_table["vocab"] = vocab; label_table["tier"] = tier.tolist()

    boot = None
    if a.metadata:
        import pandas as pd
        md = pd.read_csv(a.metadata, index_col="ecg_id")
        pid = md.loc[md["strat_fold"] == 10, "patient_id"].to_numpy()
        assert len(pid) == len(y), "metadata/test length mismatch"
        groups = {u: np.flatnonzero(pid == u) for u in np.unique(pid)}
        keys = list(groups); rng = np.random.default_rng(a.seed)
        samples = {m: [] for m in MODELS}; diffs = []
        for b in range(a.n_boot):
            idx = np.concatenate([groups[k] for k in rng.choice(keys, size=len(keys), replace=True)])
            sb = {m: scalars(data[m][0][idx], data[m][1][idx], data[m][2][idx], tier) for m in MODELS}
            for m in MODELS: samples[m].append(sb[m])
            if (b + 1) % 100 == 0: print(f"bootstrap {b+1}/{a.n_boot}")
        boot = {}
        for m in MODELS:
            boot[m] = {}
            for k in results[m]:
                vals = np.array([s[k] for s in samples[m]], dtype=float); vals = vals[np.isfinite(vals)]
                if len(vals):
                    lo, hi = np.percentile(vals, [2.5, 97.5])
                    # basic (reverse-percentile) interval is robust to the upward bias of binned ECE
                    est = results[m][k]
                    boot[m][k] = {"percentile": [float(lo), float(hi)], "basic": [float(2 * est - hi), float(2 * est - lo)],
                                  "bootstrap_mean": float(vals.mean()), "bias": float(vals.mean() - est)}
        # paired differences Mamba - ResNet for selected scalars
        boot["diff_ECG-Mamba_minus_ResNet1D"] = {}
        for k in results["ResNet1D"]:
            d = np.array([sm[k] - sr[k] for sm, sr in zip(samples["ECG-Mamba"], samples["ResNet1D"])], float)
            d = d[np.isfinite(d)]
            if len(d):
                est = results["ECG-Mamba"][k] - results["ResNet1D"][k]
                lo, hi = np.percentile(d, [2.5, 97.5])
                boot["diff_ECG-Mamba_minus_ResNet1D"][k] = {"estimate": float(est), "percentile": [float(lo), float(hi)], "basic": [float(2*est-hi), float(2*est-lo)]}

    json.dump({"point_estimates": results, "bootstrap": boot,
               "notes": ["Selective prediction uses raw probabilities unless prefixed ts.",
                         "ece = 15 equal-width bins; aece = 15 equal-mass bins; ece_rel = ECE / test prevalence;",
                         "oe_gmean = geometric mean over labels of (sum of predicted probabilities / observed positives)."]},
              open(O / "paper_analysis.json", "w"), indent=1)
    json.dump(curves, open(O / "selective_curves.json", "w"))
    json.dump(label_table, open(O / "per_label_table.json", "w"), indent=1)
    print("wrote", O)


if __name__ == "__main__":
    main()
