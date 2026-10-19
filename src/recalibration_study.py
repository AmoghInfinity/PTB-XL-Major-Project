"""
Offline post-hoc recalibration study (no GPU, no retraining).

All calibrators are fitted on VALIDATION predictions (fold 9) only and
evaluated on the TEST fold (fold 10).

Calibrators:
  none        raw sigmoid outputs
  global_ts   one temperature for all labels (reproduces temperature_scaling*.py)
  tier_ts     one temperature per frequency tier
  label_ts    one temperature per label; labels with < MIN_POS validation positives
              fall back to their tier temperature
  label_platt per-label Platt scaling (a*z + b) with L2 shrinkage toward (1, 0);
              same fallback rule

Usage:
  python src/recalibration_study.py --results results --tiers-json tiers.json --out results/paper_analysis
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import f1_score

TIERS = ("common", "medium", "rare")
CLIP = 1e-7
MIN_POS = 5
L2 = 1.0  # Platt shrinkage strength (in units of mean BCE * n)

PAIRS = {
    "ResNet1D": ("raw_calibration_validation_predictions.npz", "baseline_test_predictions.npz"),
    "ECG-Mamba": ("ecg_mamba_validation_predictions.npz", "ecg_mamba_test_predictions.npz"),
}


def load(p):
    d = np.load(p, allow_pickle=False)
    return d["y_true"].astype(np.float64), d["y_prob"].astype(np.float64), list(d["vocab"].astype(str))


def bce(y, z):
    return np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z))))


def fit_T(y, z):
    r = minimize_scalar(lambda lt: bce(y, z / np.exp(lt)), bounds=(np.log(0.05), np.log(20)),
                        method="bounded", options={"xatol": 1e-6})
    return float(np.exp(r.x))


def fit_platt(y, z):
    n = y.size
    def f(w):
        return bce(y, w[0] * z + w[1]) + L2 / n * ((w[0] - 1) ** 2 + w[1] ** 2)
    r = minimize(f, x0=np.array([1.0, 0.0]), method="L-BFGS-B", bounds=[(1e-3, 20), (-20, 20)])
    return r.x


def ece(y, p, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    return sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(n_bins) if (idx == b).any())


def evaluate(y, p, tier):
    L = y.shape[1]
    e = np.array([ece(y[:, j], p[:, j]) for j in range(L)])
    prev = y.mean(0)
    pos = y.sum(0)
    oe = p.sum(0) / np.where(pos > 0, pos, np.nan)
    nll = np.array([bce(y[:, j], logit(np.clip(p[:, j], CLIP, 1 - CLIP))) for j in range(L)])
    br = ((p - y) ** 2).mean(0)
    f1 = np.array([f1_score(y[:, j], p[:, j] >= 0.5, zero_division=0) for j in range(L)])
    out = {}
    for t in TIERS + ("macro",):
        m = np.ones(L, bool) if t == "macro" else tier == t
        o = oe[m]; o = o[np.isfinite(o) & (o > 0)]
        out[t] = {"ece": float(e[m].mean()), "ece_rel": float((e[m] / prev[m]).mean()),
                  "eo_gmean": float(np.exp(np.log(o).mean())), "nll": float(nll[m].mean()),
                  "brier": float(br[m].mean()), "f1@0.5": float(f1[m].mean())}
    # selective prediction with mean normalized entropy
    q = np.clip(p, 1e-12, 1 - 1e-12)
    H = (-(q * np.log(q) + (1 - q) * np.log(1 - q)) / np.log(2)).mean(1)
    order = np.argsort(H, kind="stable"); n = len(y)
    risk = ((p >= 0.5) != y).mean(1)
    cum = np.cumsum(risk[order]) / np.arange(1, n + 1)
    fc, fr = np.r_[0, np.arange(1, n + 1) / n], np.r_[0, cum]
    out["aurc"] = float(np.sum(np.diff(fc) * (fr[:-1] + fr[1:]) / 2))
    k = int(np.ceil(0.5 * n)); sel = order[:k]
    out["rare_retained@0.5"] = float(y[sel][:, tier == "rare"].sum() / y[:, tier == "rare"].sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--tiers-json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); R = Path(a.results); O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    T = json.load(open(a.tiers_json)); vocab = T["vocab"]; tier = np.array([T["tier_of"][v] for v in vocab])
    report = {"min_val_positives_for_label_fit": MIN_POS, "platt_l2": L2, "models": {}}
    for name, (fv, ft) in PAIRS.items():
        yv, pv, v1 = load(R / fv); yt, pt, v2 = load(R / ft)
        assert v1 == vocab == v2
        zv = logit(np.clip(pv, CLIP, 1 - CLIP)); zt = logit(np.clip(pt, CLIP, 1 - CLIP))
        vpos = yv.sum(0)
        params = {}
        Tg = fit_T(yv, zv); params["global_ts"] = Tg
        Tt = {t: fit_T(yv[:, tier == t], zv[:, tier == t]) for t in TIERS}; params["tier_ts"] = Tt
        Tl = np.array([fit_T(yv[:, j], zv[:, j]) if vpos[j] >= MIN_POS else Tt[tier[j]] for j in range(len(vocab))])
        params["label_ts"] = dict(zip(vocab, Tl.round(4).tolist()))
        W = []
        for j in range(len(vocab)):
            if vpos[j] >= MIN_POS:
                W.append(fit_platt(yv[:, j], zv[:, j]))
            else:
                W.append(np.array([1 / Tt[tier[j]], 0.0]))
        W = np.array(W); params["label_platt"] = {v: [round(float(a_), 4), round(float(b_), 4)] for v, (a_, b_) in zip(vocab, W)}
        cal = {
            "none": pt,
            "global_ts": expit(zt / Tg),
            "tier_ts": expit(zt / np.array([Tt[t] for t in tier])),
            "label_ts": expit(zt / Tl),
            "label_platt": expit(zt * W[:, 0] + W[:, 1]),
        }
        report["models"][name] = {"params": params, "test": {k: evaluate(yt, p, tier) for k, p in cal.items()},
                                  "n_labels_fallback": int((vpos < MIN_POS).sum())}
        np.savez_compressed(O / f"{name.lower().replace('-', '_')}_recalibrated_test.npz",
                            y_true=yt, vocab=np.array(vocab), **{k: v for k, v in cal.items()})
    json.dump(report, open(O / "recalibration_study.json", "w"), indent=1)
    for name, r in report["models"].items():
        print(f"\n{name}  global T={r['params']['global_ts']:.4f}  tier T={ {k: round(v,3) for k,v in r['params']['tier_ts'].items()} }  fallback labels={r['n_labels_fallback']}")
        print(f"{'method':12s}" + "".join(f"{t[:3]}:ECE  {t[:3]}:rel  {t[:3]}:E/O  " for t in TIERS) + " macroNLL  AURC   rare@.5")
        for k, e in r["test"].items():
            print(f"{k:12s}" + "".join(f"{e[t]['ece']:.5f} {e[t]['ece_rel']:.3f} {e[t]['eo_gmean']:.3f}   " for t in TIERS) + f" {e['macro']['nll']:.5f} {e['aurc']:.5f} {e['rare_retained@0.5']:.3f}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Bootstrap of recalibration differences + Platt hyper-parameter sensitivity.
#   python src/recalibration_study.py ... then:
#   python -c "import recalibration_study as r; r.bootstrap_main()" --results ... (see CLI below)
# ---------------------------------------------------------------------------
def _fast_metrics(y, p, tier, n_bins=15):
    """Vectorised tier metrics used inside the bootstrap."""
    n, L = y.shape
    idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    e = np.zeros(L)
    for b in range(n_bins):
        m = idx == b
        cnt = m.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            gap = np.abs((y * m).sum(0) - (p * m).sum(0))
        e += np.where(cnt > 0, gap, 0.0) / n
    pos = y.sum(0); prev = pos / n
    z = logit(np.clip(p, CLIP, 1 - CLIP))
    nll = (np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))).mean(0)
    out = {"nll_macro": nll.mean()}
    for t in TIERS:
        m = (tier == t) & (pos > 0)
        out[f"ece_{t}"] = e[tier == t].mean()
        out[f"ece_rel_{t}"] = (e[m] / prev[m]).mean()
        out[f"log_eo_{t}"] = np.log(p.sum(0)[m] / pos[m]).mean()
    q = np.clip(p, 1e-12, 1 - 1e-12)
    H = (-(q * np.log(q) + (1 - q) * np.log(1 - q))).mean(1)
    order = np.argsort(H, kind="stable")
    risk = ((p >= 0.5) != y).mean(1)
    cum = np.cumsum(risk[order]) / np.arange(1, n + 1)
    out["aurc"] = (cum.sum() - cum[-1] / 2 + 0) / n  # trapezoid with (0,0) start, matches main script up to O(1/n)
    return out


def bootstrap_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--recal-dir", required=True, help="folder with *_recalibrated_test.npz")
    ap.add_argument("--tiers-json", required=True)
    ap.add_argument("--metadata", help="ptbxl_database.csv -> patient-cluster bootstrap; otherwise record-level")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    T = json.load(open(a.tiers_json)); tier = np.array([T["tier_of"][v] for v in T["vocab"]])
    D = Path(a.recal_dir); res = {}
    groups = None
    if a.metadata:
        import pandas as pd
        md = pd.read_csv(a.metadata, index_col="ecg_id")
        pid = md.loc[md["strat_fold"] == 10, "patient_id"].to_numpy()
        groups = [np.flatnonzero(pid == u) for u in np.unique(pid)]
    for name in ("resnet1d", "ecg_mamba"):
        d = np.load(D / f"{name}_recalibrated_test.npz")
        y = d["y_true"]; methods = ["none", "global_ts", "tier_ts", "label_ts", "label_platt"]
        P = {m: d[m] for m in methods}
        rng = np.random.default_rng(a.seed); n = len(y)
        obs = {m: _fast_metrics(y, P[m], tier) for m in methods}
        diffs = {(m, ref): {k: [] for k in obs[m]} for m in methods for ref in ("none", "global_ts") if m != ref}
        for _ in range(a.n_boot):
            if groups is None:
                idx = rng.integers(0, n, n)
            else:
                idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            mb = {m: _fast_metrics(y[idx], P[m][idx], tier) for m in methods}
            for (m, ref), store in diffs.items():
                for k in store:
                    store[k].append(mb[m][k] - mb[ref][k])
        out = {}
        for (m, ref), store in diffs.items():
            out[f"{m}_minus_{ref}"] = {}
            for k, vals in store.items():
                vals = np.asarray(vals); vals = vals[np.isfinite(vals)]
                est = obs[m][k] - obs[ref][k]
                lo, hi = np.percentile(vals, [2.5, 97.5])
                out[f"{m}_minus_{ref}"][k] = {"estimate": float(est), "percentile_ci": [float(lo), float(hi)],
                                               "basic_ci": [float(2 * est - hi), float(2 * est - lo)]}
        res[name] = {"observed": {m: {k: float(v) for k, v in o.items()} for m, o in obs.items()}, "differences": out}
    res["resampling_unit"] = "patient" if groups is not None else "record (approximation; patient-level requires metadata)"
    res["n_boot"] = a.n_boot
    json.dump(res, open(D / "recalibration_bootstrap.json", "w"), indent=1)
    print("resampling unit:", res["resampling_unit"])
    for name in ("resnet1d", "ecg_mamba"):
        for comp in ("label_platt_minus_global_ts", "tier_ts_minus_global_ts", "global_ts_minus_none", "label_platt_minus_none"):
            r = res[name]["differences"][comp]
            print(name, comp, " | ".join(f"{k}={v['estimate']:+.5f} [{v['basic_ci'][0]:+.5f},{v['basic_ci'][1]:+.5f}]" for k, v in r.items() if k in ("nll_macro", "ece_common", "ece_rel_rare", "log_eo_rare", "aurc")))
