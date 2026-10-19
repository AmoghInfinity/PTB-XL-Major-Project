"""Aggregate multi-seed predictions into mean +/- SD tables for the manuscript.

    python src/aggregate_seeds.py --seeds-dir results/seeds --tiers-json tiers.json --out results/seeds/aggregate.json

Expects results/seeds/s<k>/<tag>_{val,test}.npz written by predict_split.py.
For every seed and model it computes test discrimination, calibration under the five
post-hoc calibrators (fitted on that seed's validation predictions only), and
selective-prediction retention. It then reports mean, SD and the paired
ECG-Mamba minus ResNet1D difference across seeds.
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recalibration_study as rs

TIERS = ("common", "medium", "rare")


def discrim(y, p, tier):
    L = y.shape[1]
    au = np.array([roc_auc_score(y[:, j], p[:, j]) if 0 < y[:, j].sum() < len(y) else np.nan for j in range(L)])
    ap = np.array([average_precision_score(y[:, j], p[:, j]) if y[:, j].sum() > 0 else np.nan for j in range(L)])
    f1 = np.array([f1_score(y[:, j], p[:, j] >= 0.5, zero_division=0) for j in range(L)])
    o = {"auroc_macro": np.nanmean(au), "auprc_macro": np.nanmean(ap), "f1_macro": f1.mean()}
    for t in TIERS:
        m = tier == t
        o[f"auroc_{t}"] = np.nanmean(au[m]); o[f"auprc_{t}"] = np.nanmean(ap[m]); o[f"f1_{t}"] = f1[m].mean()
    return o


def calibrators(yv, pv, pt, tier):
    zv = logit(np.clip(pv, rs.CLIP, 1 - rs.CLIP)); zt = logit(np.clip(pt, rs.CLIP, 1 - rs.CLIP))
    vpos = yv.sum(0)
    Tg = rs.fit_T(yv, zv)
    Tt = {t: rs.fit_T(yv[:, tier == t], zv[:, tier == t]) for t in TIERS}
    Tvec = np.array([Tt[t] for t in tier])
    W = np.array([rs.fit_platt(yv[:, j], zv[:, j]) if vpos[j] >= rs.MIN_POS else [1 / Tvec[j], 0.0]
                  for j in range(yv.shape[1])])
    return {"none": pt, "global_ts": expit(zt / Tg), "tier_ts": expit(zt / Tvec),
            "label_platt": expit(zt * W[:, 0] + W[:, 1])}, Tg


def flatten(prefix, d, out):
    for k, v in d.items():
        if isinstance(v, dict):
            flatten(f"{prefix}{k}.", v, out)
        else:
            out[f"{prefix}{k}"] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-dir", required=True); ap.add_argument("--tiers-json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T = json.load(open(a.tiers_json)); vocab = T["vocab"]; tier = np.array([T["tier_of"][v] for v in vocab])
    per = {}
    for sd in sorted(Path(a.seeds_dir).glob("s*")):
        if not sd.is_dir() or not re.fullmatch(r"s\d+", sd.name):
            continue
        for f in sorted(sd.glob("*_test.npz")):
            tag = f.name[:-len("_test.npz")]
            v = sd / f"{tag}_val.npz"
            if not v.exists():
                continue
            dt, dv = np.load(f), np.load(v)
            assert list(dt["vocab"].astype(str)) == vocab
            yt, pt = dt["y_true"].astype(float), dt["y_prob"].astype(float)
            yv, pv = dv["y_true"].astype(float), dv["y_prob"].astype(float)
            cal, Tg = calibrators(yv, pv, pt, tier)
            row = {"global_T": Tg, "best_epoch": float(dt["epoch"])}
            row.update({k: float(x) for k, x in discrim(yt, pt, tier).items()})
            for name, p in cal.items():
                flatten(f"{name}.", rs.evaluate(yt, p, tier), row)
            per.setdefault(tag, {})[sd.name] = row
    summary = {}
    for tag, seeds in per.items():
        keys = next(iter(seeds.values())).keys()
        summary[tag] = {"n_seeds": len(seeds), "seeds": sorted(seeds),
                        **{k: {"mean": float(np.mean([s[k] for s in seeds.values()])),
                               "sd": float(np.std([s[k] for s in seeds.values()], ddof=1)) if len(seeds) > 1 else None}
                           for k in keys}}
    paired = {}
    if "resnet1d_bce" in per and "ecg_mamba_bce" in per:
        common = sorted(set(per["resnet1d_bce"]) & set(per["ecg_mamba_bce"]))
        for k in per["resnet1d_bce"][common[0]]:
            d = np.array([per["ecg_mamba_bce"][s][k] - per["resnet1d_bce"][s][k] for s in common])
            paired[k] = {"mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)) if len(d) > 1 else None,
                         "n_mamba_higher": int((d > 0).sum()), "n_seeds": len(d)}
    json.dump({"per_seed": per, "summary": summary, "paired_ecg_mamba_minus_resnet1d": paired}, open(a.out, "w"), indent=1)
    show = ["auroc_macro", "auprc_macro", "f1_macro", "auroc_rare", "none.macro.nll", "label_platt.macro.nll",
            "none.common.ece", "global_ts.rare.eo_gmean", "label_platt.rare.eo_gmean", "none.aurc", "none.rare_retained@0.5"]
    for tag, s in summary.items():
        print(f"\n{tag} (n={s['n_seeds']})")
        for k in show:
            if k in s:
                sdv = s[k]["sd"]; print(f"  {k:28s} {s[k]['mean']:.4f} +/- {sdv if sdv is None else round(sdv, 4)}")
    if paired:
        print("\nPaired ECG-Mamba - ResNet1D across seeds")
        for k in show:
            if k in paired:
                p = paired[k]; print(f"  {k:28s} {p['mean_diff']:+.4f} (sd {p['sd_diff']}) Mamba higher in {p['n_mamba_higher']}/{p['n_seeds']}")


if __name__ == "__main__":
    main()
