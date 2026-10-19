"""Sensitivity of per-label Platt scaling to its two a-priori settings (MIN_POS, L2).
Reports TEST metrics for every setting; nothing is selected on test.
Usage: python src/platt_sensitivity.py --results results --tiers-json tiers.json --out results/paper_analysis"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
sys.path.insert(0, str(Path(__file__).resolve().parent))
import recalibration_study as rs

ap = argparse.ArgumentParser(); ap.add_argument("--results"); ap.add_argument("--tiers-json"); ap.add_argument("--out")
a = ap.parse_args(); R = Path(a.results)
T = json.load(open(a.tiers_json)); vocab = T["vocab"]; tier = np.array([T["tier_of"][v] for v in vocab])
rows = []
for name, (fv, ft) in rs.PAIRS.items():
    yv, pv, _ = rs.load(R / fv); yt, pt, _ = rs.load(R / ft)
    zv = logit(np.clip(pv, rs.CLIP, 1 - rs.CLIP)); zt = logit(np.clip(pt, rs.CLIP, 1 - rs.CLIP))
    vpos = yv.sum(0)
    Tt = {t: rs.fit_T(yv[:, tier == t], zv[:, tier == t]) for t in rs.TIERS}
    for min_pos in (3, 5, 10):
        for l2 in (0.1, 1.0, 10.0):
            rs.L2 = l2
            W = np.array([rs.fit_platt(yv[:, j], zv[:, j]) if vpos[j] >= min_pos else [1 / Tt[tier[j]], 0.0] for j in range(len(vocab))])
            e = rs.evaluate(yt, expit(zt * W[:, 0] + W[:, 1]), tier)
            rows.append({"model": name, "min_pos": min_pos, "l2": l2, "nll_macro": e["macro"]["nll"],
                         "ece_common": e["common"]["ece"], "ece_rel_rare": e["rare"]["ece_rel"],
                         "eo_rare": e["rare"]["eo_gmean"], "aurc": e["aurc"]})
            r = rows[-1]; print(f"{name:10s} min_pos={min_pos:2d} l2={l2:5.1f}  NLL={r['nll_macro']:.5f} ECEc={r['ece_common']:.5f} relECEr={r['ece_rel_rare']:.3f} E/Or={r['eo_rare']:.3f} AURC={r['aurc']:.5f}")
json.dump(rows, open(Path(a.out) / "platt_sensitivity.json", "w"), indent=1)
