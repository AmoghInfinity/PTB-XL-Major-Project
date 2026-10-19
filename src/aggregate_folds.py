"""Pool the 10-fold rotation (each PTB-XL fold used once as the test fold) and analyse it.

Run from the project root after run_folds.ps1:
    python src/aggregate_folds.py --folds-dir results/folds --tiers-json tiers.json --out results/folds/fold_rotation.json

Expects results/folds/f<k>/<tag>_{val,test}.npz written by predict_split.py
(tags: resnet1d_bce, ecg_mamba_bce). For every rotation, all calibrators are fitted on
that rotation's OWN validation fold only and applied to its test fold; the 10 test folds
are then pooled (every PTB-XL record appears exactly once). Folds are patient-disjoint,
so the pooled set can be bootstrapped by patient.

Outputs:
  fold_rotation.json          pooled + per-fold metrics, bootstrap CIs
  pooled_<tag>.npz            pooled test predictions under every calibrator (for figures)
  pooled_per_label.json       per-label AUROC/AUPRC and positives on the pooled set
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recalibration_study as rs          # fit_T, fit_platt, evaluate, _fast_metrics
from aggregate_seeds import calibrators, discrim

TIERS = ("common", "medium", "rare")
COVS = (0.9, 0.8, 0.7, 0.6, 0.5)


def fast_auroc(y, p):
    """Per-label AUROC via ranks (ties averaged); NaN when a label has no positives or no negatives."""
    r = rankdata(p, axis=0)
    n1 = y.sum(0); n0 = len(y) - n1
    with np.errstate(invalid="ignore", divide="ignore"):
        au = ((r * y).sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    au[(n1 == 0) | (n0 == 0)] = np.nan
    return au


def retention(y, p, tier):
    """Tier retention and accepted-set macro-F1 under mean-entropy abstention."""
    n = len(y); q = np.clip(p, 1e-12, 1 - 1e-12)
    H = (-(q * np.log(q) + (1 - q) * np.log(1 - q))).mean(1)
    order = np.argsort(H, kind="stable"); pred = p >= 0.5; yb = y.astype(bool)
    out = {}
    for c in COVS:
        sel = order[: int(np.ceil(c * n))]
        for t in TIERS:
            m = tier == t
            out[f"rho_{t}@{c}"] = float(yb[sel][:, m].sum() / max(yb[:, m].sum(), 1))
        ys, ps = yb[sel], pred[sel]
        tp = (ys & ps).sum(0); fp = (~ys & ps).sum(0); fn = (ys & ~ps).sum(0)
        den = 2 * tp + fp + fn
        out[f"macro_f1@{c}"] = float(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0).mean())
        out[f"risk@{c}"] = float((ps != ys).mean())
    return out


def boot_stats(y, P, tier):
    """Scalar statistics recomputed inside each bootstrap replicate (P: dict calibrator -> probs)."""
    o = {}
    for k, p in P.items():
        for name, v in rs._fast_metrics(y, p, tier).items():
            o[f"{k}.{name}"] = float(v)
    au = fast_auroc(y, P["none"])
    o["auroc_macro"] = float(np.nanmean(au))
    for t in TIERS:
        o[f"auroc_{t}"] = float(np.nanmean(au[tier == t]))
    r = retention(y, P["none"], tier)
    for t in TIERS:
        o[f"rho_{t}@0.5"] = r[f"rho_{t}@0.5"]; o[f"rho_{t}@0.8"] = r[f"rho_{t}@0.8"]
    pred = P["none"] >= 0.5; yb = y.astype(bool)
    tp = (yb & pred).sum(0); fp = (~yb & pred).sum(0); fn = (yb & ~pred).sum(0); den = 2 * tp + fp + fn
    o["macro_f1@1.0"] = float(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0).mean())
    o["macro_f1@0.5"] = r["macro_f1@0.5"]
    return o


def ci(vals, est, kind):
    v = np.asarray(vals, float); v = v[np.isfinite(v)]
    lo, hi = np.percentile(v, [2.5, 97.5])
    return [float(2 * est - hi), float(2 * est - lo)] if kind == "basic" else [float(lo), float(hi)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds-dir", required=True); ap.add_argument("--tiers-json", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    T = json.load(open(a.tiers_json)); vocab = T["vocab"]; tier = np.array([T["tier_of"][v] for v in vocab])
    D = Path(a.folds_dir); out_dir = Path(a.out).parent; out_dir.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted([d for d in D.glob("f*") if d.is_dir() and re.fullmatch(r"f\d+", d.name)], key=lambda d: int(d.name[1:]))
    tags = sorted({f.name[:-len("_test.npz")] for d in fold_dirs for f in d.glob("*_test.npz")})
    result = {"folds": [d.name for d in fold_dirs], "tags": tags, "pooled": {}, "per_fold": {}, "temperatures": {}}
    pooled = {}

    for tag in tags:
        ys, ids, pids, Pc = [], [], [], {}
        perfold, temps = [], []
        for d in fold_dirs:
            ft, fv = d / f"{tag}_test.npz", d / f"{tag}_val.npz"
            if not (ft.exists() and fv.exists()):
                print(f"  missing {tag} in {d.name}; skipped"); continue
            dt, dv = np.load(ft), np.load(fv)
            assert list(dt["vocab"].astype(str)) == vocab, "vocabulary mismatch"
            yt, pt = dt["y_true"].astype(float), dt["y_prob"].astype(float)
            yv, pv = dv["y_true"].astype(float), dv["y_prob"].astype(float)
            cal, Tg = calibrators(yv, pv, pt, tier)
            Tt = {t: rs.fit_T(yv[:, tier == t], rs.logit(np.clip(pv[:, tier == t], rs.CLIP, 1 - rs.CLIP))) for t in TIERS}
            temps.append({"fold": d.name, "global": Tg, **Tt})
            for k, p in cal.items():
                Pc.setdefault(k, []).append(p)
            ys.append(yt); ids.append(dt["ecg_id"]); pids.append(dt["patient_id"])
            row = {"fold": d.name, "n": int(len(yt)), "rare_pos": int(yt[:, tier == "rare"].sum())}
            row.update({k: float(v) for k, v in discrim(yt, pt, tier).items()})
            for k, p in cal.items():
                e = rs.evaluate(yt, p, tier)
                for t in TIERS + ("macro",):
                    for m_, v in e[t].items():
                        row[f"{k}.{t}.{m_}"] = float(v)
                row[f"{k}.aurc"] = e["aurc"]
            row.update({f"none.{k}": v for k, v in retention(yt, pt, tier).items()})
            perfold.append(row)
        y = np.concatenate(ys); ecg = np.concatenate(ids); pid = np.concatenate(pids)
        assert len(np.unique(ecg)) == len(ecg), "a record appears in more than one test fold"
        P = {k: np.concatenate(v) for k, v in Pc.items()}
        pooled[tag] = (y, P, pid, ecg)
        np.savez_compressed(out_dir / f"pooled_{tag}.npz", y_true=y, ecg_id=ecg, patient_id=pid, vocab=np.array(vocab), **P)

        # pooled point estimates
        pr = {"n_records": int(len(y)), "n_patients": int(len(np.unique(pid))),
              "positives_per_tier": {t: int(y[:, tier == t].sum()) for t in TIERS}}
        pr.update({k: float(v) for k, v in discrim(y, P["none"], tier).items()})
        for k, p in P.items():
            e = rs.evaluate(y, p, tier)
            pr[k] = {t: e[t] for t in TIERS + ("macro",)}
            pr[k]["aurc"] = e["aurc"]
        pr["selective_none"] = retention(y, P["none"], tier)
        result["pooled"][tag] = pr
        keys = [k for k in perfold[0] if k not in ("fold",)]
        result["per_fold"][tag] = {"rows": perfold, "summary": {
            k: {"mean": float(np.mean([r[k] for r in perfold])), "sd": float(np.std([r[k] for r in perfold], ddof=1)),
                "min": float(np.min([r[k] for r in perfold])), "max": float(np.max([r[k] for r in perfold]))} for k in keys}}
        result["temperatures"][tag] = temps

    # per-label pooled table
    lab = {"vocab": vocab, "tier": tier.tolist()}
    for tag, (y, P, _, _) in pooled.items():
        lab["positives"] = y.sum(0).astype(int).tolist()
        lab[tag] = {"auroc": [None if not np.isfinite(x) else float(x) for x in fast_auroc(y, P["none"])]}
    json.dump(lab, open(out_dir / "pooled_per_label.json", "w"), indent=1)

    # patient-cluster bootstrap on the pooled set (paired across calibrators and, if aligned, across models)
    rng = np.random.default_rng(a.seed)
    ref = tags[0]; y0, _, pid0, ecg0 = pooled[ref]
    aligned = all(np.array_equal(pooled[t][3], ecg0) for t in tags)
    groups = [np.flatnonzero(pid0 == u) for u in np.unique(pid0)]
    obs = {t: boot_stats(pooled[t][0], pooled[t][1], tier) for t in tags}
    samples = {t: [] for t in tags}
    for b in range(a.n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        for t in tags:
            y_, P_, _, _ = pooled[t]
            if not aligned and t != ref:
                continue
            samples[t].append(boot_stats(y_[idx], {k: v[idx] for k, v in P_.items()}, tier))
        if (b + 1) % 100 == 0:
            print(f"bootstrap {b + 1}/{a.n_boot}")
    comps = {
        "globalTS_minus_none.log_eo_rare": ("global_ts.log_eo_rare", "none.log_eo_rare", "basic"),
        "globalTS_minus_none.ece_rel_rare": ("global_ts.ece_rel_rare", "none.ece_rel_rare", "basic"),
        "platt_minus_globalTS.nll_macro": ("label_platt.nll_macro", "global_ts.nll_macro", "basic"),
        "platt_minus_globalTS.ece_common": ("label_platt.ece_common", "global_ts.ece_common", "basic"),
        "platt_minus_globalTS.ece_rel_rare": ("label_platt.ece_rel_rare", "global_ts.ece_rel_rare", "basic"),
        "platt_minus_none.nll_macro": ("label_platt.nll_macro", "none.nll_macro", "basic"),
        "tierTS_minus_globalTS.log_eo_rare": ("tier_ts.log_eo_rare", "global_ts.log_eo_rare", "basic"),
    }
    boot = {}
    for t in tags:
        if not samples[t]:
            continue
        S = samples[t]; boot[t] = {}
        for name, (x1, x0, kind) in comps.items():
            est = obs[t][x1] - obs[t][x0]
            boot[t][name] = {"estimate": est, "ci": ci([s[x1] - s[x0] for s in S], est, kind), "ci_type": kind}
        for k in ("rho_common@0.5", "rho_medium@0.5", "rho_rare@0.5", "rho_rare@0.8", "auroc_macro", "auroc_rare",
                  "macro_f1@1.0", "macro_f1@0.5", "none.log_eo_rare", "global_ts.log_eo_rare", "label_platt.log_eo_rare"):
            boot[t][k] = {"estimate": obs[t][k], "ci": ci([s[k] for s in S], obs[t][k], "percentile"), "ci_type": "percentile"}
    if aligned and len(tags) == 2 and all(samples[t] for t in tags):
        a_, b_ = "ecg_mamba_bce", "resnet1d_bce"
        if a_ in tags and b_ in tags:
            boot["mamba_minus_resnet"] = {}
            for k in ("auroc_macro", "auroc_rare", "label_platt.nll_macro", "none.nll_macro", "rho_rare@0.5", "none.aurc"):
                est = obs[a_][k] - obs[b_][k]
                boot["mamba_minus_resnet"][k] = {"estimate": est, "ci": ci([x[k] - y_[k] for x, y_ in zip(samples[a_], samples[b_])], est, "percentile")}
    result["bootstrap"] = {"unit": "patient", "n_boot": a.n_boot, "seed": a.seed, "results": boot}
    json.dump(result, open(a.out, "w"), indent=1)

    # console summary
    for tag in tags:
        pr = result["pooled"][tag]
        print(f"\n=== {tag}: {pr['n_records']} records, {pr['n_patients']} patients, positives {pr['positives_per_tier']}")
        print(f"  AUROC macro {pr['auroc_macro']:.4f}  common {pr['auroc_common']:.4f}  medium {pr['auroc_medium']:.4f}  rare {pr['auroc_rare']:.4f}  AUPRC {pr['auprc_macro']:.4f}")
        for k in ("none", "global_ts", "tier_ts", "label_platt"):
            e = pr[k]
            print(f"  {k:12s} commonECE {e['common']['ece']:.5f}  rareRelECE {e['rare']['ece_rel']:.3f}  rareE/O {e['rare']['eo_gmean']:.3f}  NLL {e['macro']['nll']:.5f}  AURC {e['aurc']:.5f}")
        s = pr["selective_none"]
        print("  retention @0.5: " + "  ".join(f"{t} {s[f'rho_{t}@0.5']:.3f}" for t in TIERS) + f"   macroF1 {s['macro_f1@0.5']:.3f}")
        if tag in boot:
            for k, v in boot[tag].items():
                print(f"    {k:38s} {v['estimate']:+.4f}  [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")
    if "mamba_minus_resnet" in boot:
        print("\nECG-Mamba minus ResNet1D (pooled, patient bootstrap):")
        for k, v in boot["mamba_minus_resnet"].items():
            print(f"    {k:28s} {v['estimate']:+.4f}  [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()