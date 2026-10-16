"""Paired patient-level comparison of ResNet1D and ECG-Mamba."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
BASELINE_PATH = RESULTS / "baseline_test_predictions.npz"
MAMBA_PATH = RESULTS / "ecg_mamba_test_predictions.npz"
OUTPUT_PATH = RESULTS / "paired_model_comparison.json"

N_BOOTSTRAPS = 1000
SEED = 42
THRESHOLD = 0.5
ECE_BINS = 15


def load_predictions(path):
    with np.load(path, allow_pickle=False) as data:
        return (
            data["y_true"].astype(np.float64),
            data["y_prob"].astype(np.float64),
            data["vocab"].astype(str),
        )


def safe_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def mean_or_none(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else float("nan")


def per_label_metric(function, y_true, y_prob):
    values = []

    for j in range(y_true.shape[1]):
        target = y_true[:, j]
        probability = y_prob[:, j]

        if function is roc_auc_score and np.unique(target).size < 2:
            values.append(float("nan"))
            continue

        if function is average_precision_score and not np.any(target == 1):
            values.append(float("nan"))
            continue

        try:
            values.append(function(target, probability))
        except ValueError:
            values.append(float("nan"))

    return np.asarray(values, dtype=np.float64)


def per_label_mcc(y_true, y_pred):
    targets = y_true.astype(bool)
    predictions = y_pred.astype(bool)

    tp = np.sum(targets & predictions, axis=0, dtype=np.float64)
    tn = np.sum(~targets & ~predictions, axis=0, dtype=np.float64)
    fp = np.sum(~targets & predictions, axis=0, dtype=np.float64)
    fn = np.sum(targets & ~predictions, axis=0, dtype=np.float64)

    numerator = tp * tn - fp * fn
    denominator = np.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )

    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def ece_binary(y_true, y_prob, bins=ECE_BINS):
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0

    for i in range(bins):
        if i == bins - 1:
            mask = (y_prob >= edges[i]) & (y_prob <= edges[i + 1])
        else:
            mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1])

        if np.any(mask):
            result += np.mean(mask) * abs(
                np.mean(y_prob[mask]) - np.mean(y_true[mask])
            )

    return float(result)


def weighted_metric(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)

    if not np.any(valid):
        return None

    return safe_float(np.average(values[valid], weights=weights[valid]))


def calculate_metrics(y_true, y_prob, tier_masks):
    eps = 1e-7
    probabilities = np.clip(y_prob, eps, 1.0 - eps)
    predictions = (y_prob >= THRESHOLD).astype(np.int32)
    targets = y_true.astype(np.int32)

    support = np.sum(targets, axis=0).astype(np.float64)
    label_weights = support / support.sum() if support.sum() else np.zeros_like(support)

    label_auc = per_label_metric(roc_auc_score, targets, probabilities)
    label_ap = per_label_metric(average_precision_score, targets, probabilities)
    label_precision = per_label_metric(
        lambda y, p: precision_score(
            y, p >= THRESHOLD, zero_division=0
        ), targets, probabilities
    )
    label_recall = per_label_metric(
        lambda y, p: recall_score(
            y, p >= THRESHOLD, zero_division=0
        ), targets, probabilities
    )
    label_f1 = per_label_metric(
        lambda y, p: f1_score(
            y, p >= THRESHOLD, zero_division=0
        ), targets, probabilities
    )
    label_specificity = per_label_metric(
        lambda y, p: recall_score(
            1 - y, 1 - (p >= THRESHOLD), zero_division=0
        ), targets, probabilities
    )
    label_mcc = per_label_mcc(targets, predictions)
    label_brier = np.mean((probabilities - targets) ** 2, axis=0)
    label_ece = np.asarray([
        ece_binary(targets[:, j], probabilities[:, j])
        for j in range(targets.shape[1])
    ])

    metrics = {
        "auroc_macro": mean_or_none(label_auc),
        "auroc_micro": safe_float(roc_auc_score(
            targets.ravel(), probabilities.ravel()
        )),
        "auroc_weighted": weighted_metric(label_auc, label_weights),
        "auprc_macro": mean_or_none(label_ap),
        "auprc_micro": safe_float(average_precision_score(
            targets.ravel(), probabilities.ravel()
        )),
        "auprc_weighted": weighted_metric(label_ap, label_weights),
        "precision_macro": mean_or_none(label_precision),
        "precision_micro": safe_float(precision_score(
            targets.ravel(), predictions.ravel(), zero_division=0
        )),
        "recall_macro": mean_or_none(label_recall),
        "recall_micro": safe_float(recall_score(
            targets.ravel(), predictions.ravel(), zero_division=0
        )),
        "f1_macro": mean_or_none(label_f1),
        "f1_micro": safe_float(f1_score(
            targets.ravel(), predictions.ravel(), zero_division=0
        )),
        "f1_weighted": safe_float(np.average(
            label_f1, weights=label_weights
        )) if np.sum(label_weights) else None,
        "specificity_macro": mean_or_none(label_specificity),
        "balanced_accuracy_macro": mean_or_none(
            (label_recall + label_specificity) / 2.0
        ),
        "mcc_macro": mean_or_none(label_mcc),
        "mcc_micro": safe_float(matthews_corrcoef(
            targets.ravel(), predictions.ravel()
        )),
        "hamming_loss": safe_float(hamming_loss(targets, predictions)),
        "jaccard_macro": safe_float(jaccard_score(
            targets, predictions, average="macro", zero_division=0
        )),
        "jaccard_micro": safe_float(jaccard_score(
            targets, predictions, average="micro", zero_division=0
        )),
        "log_loss": safe_float(log_loss(
            targets.ravel(), probabilities.ravel(), labels=[0, 1]
        )),
        "brier_macro": safe_float(np.mean(label_brier)),
        "ece_macro": safe_float(np.mean(label_ece)),
    }

    for tier, mask in tier_masks.items():
        if not np.any(mask):
            continue
        metrics[f"auroc_{tier}"] = mean_or_none(label_auc[mask])
        metrics[f"auprc_{tier}"] = mean_or_none(label_ap[mask])
        metrics[f"f1_{tier}"] = mean_or_none(label_f1[mask])
        metrics[f"ece_{tier}"] = mean_or_none(label_ece[mask])
        metrics[f"brier_{tier}"] = mean_or_none(label_brier[mask])

    return metrics


def patient_bootstrap_indices(patient_ids, rng):
    unique_patients = np.unique(patient_ids)
    sampled_patients = rng.choice(
        unique_patients, size=len(unique_patients), replace=True
    )
    return np.concatenate([
        np.flatnonzero(patient_ids == patient_id)
        for patient_id in sampled_patients
    ])


def interval(values):
    values = np.asarray(
        [v for v in values if v is not None and np.isfinite(v)],
        dtype=np.float64
    )
    if not len(values):
        return {"lower": None, "upper": None, "valid_bootstraps": 0}
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "lower": float(low),
        "upper": float(high),
        "valid_bootstraps": int(len(values)),
    }


def main():
    for path in (BASELINE_PATH, MAMBA_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file: {path}")

    y_base, p_base, vocab_base = load_predictions(BASELINE_PATH)
    y_mamba, p_mamba, vocab_mamba = load_predictions(MAMBA_PATH)

    if not np.array_equal(y_base, y_mamba):
        raise ValueError("Model targets do not match.")
    if not np.array_equal(vocab_base, vocab_mamba):
        raise ValueError("Model label vocabularies do not match.")

    sys.path.insert(0, str(ROOT / "src"))
    from data.dataset import PTBXLDataset

    dataset = PTBXLDataset(split="test")
    patient_ids = dataset.df["patient_id"].to_numpy()

    if len(patient_ids) != len(y_base):
        raise ValueError("Patient IDs do not align with predictions.")
    if not np.array_equal(np.asarray(dataset.vocab).astype(str), vocab_base):
        raise ValueError("Dataset vocabulary order differs from predictions.")

    tier_masks = {
        tier: np.asarray(dataset.tier_mask(tier), dtype=bool)
        for tier in ("common", "medium", "rare")
    }

    observed_base = calculate_metrics(y_base, p_base, tier_masks)
    observed_mamba = calculate_metrics(y_base, p_mamba, tier_masks)
    metric_names = sorted(set(observed_base) & set(observed_mamba))

    base_boot = {name: [] for name in metric_names}
    mamba_boot = {name: [] for name in metric_names}
    diff_boot = {name: [] for name in metric_names}

    rng = np.random.default_rng(SEED)

    for iteration in range(N_BOOTSTRAPS):
        indices = patient_bootstrap_indices(patient_ids, rng)
        y = y_base[indices]

        base_metrics = calculate_metrics(y, p_base[indices], tier_masks)
        mamba_metrics = calculate_metrics(y, p_mamba[indices], tier_masks)

        for name in metric_names:
            a = base_metrics.get(name)
            b = mamba_metrics.get(name)

            base_boot[name].append(a)
            mamba_boot[name].append(b)

            if a is not None and b is not None:
                diff_boot[name].append(b - a)
            else:
                diff_boot[name].append(None)

        if (iteration + 1) % 100 == 0:
            print(f"Completed {iteration + 1}/{N_BOOTSTRAPS} bootstraps")

    RESULTS.mkdir(parents=True, exist_ok=True)

    bootstrap_samples = {}
    for metric_name in metric_names:
        for model_name, samples in (
            ("resnet1d", base_boot[metric_name]),
            ("ecg_mamba", mamba_boot[metric_name]),
            ("difference", diff_boot[metric_name]),
        ):
            bootstrap_samples[f"{model_name}__{metric_name}"] = np.asarray(
                [
                    np.nan if value is None else value
                    for value in samples
                ],
                dtype=np.float64,
            )

    bootstrap_path = RESULTS / "paired_model_bootstrap_samples.npz"
    np.savez_compressed(bootstrap_path, **bootstrap_samples)
    print(f"Saved bootstrap samples: {bootstrap_path}")

    output = {
        "method": "paired_patient_cluster_bootstrap",
        "n_bootstraps": N_BOOTSTRAPS,
        "random_seed": SEED,
        "threshold": THRESHOLD,
        "ece_bins": ECE_BINS,
        "test_recordings": int(len(patient_ids)),
        "unique_patients": int(len(np.unique(patient_ids))),
        "models": {
            "resnet1d": {},
            "ecg_mamba": {},
        },
        "differences_ecg_mamba_minus_resnet1d": {},
        "notes": [
            "Both models use identical test targets and label vocabulary.",
            "The same patient-cluster bootstrap samples are used for both models.",
            "Positive differences favor ECG-Mamba only for metrics where higher is better; negative differences favor ECG-Mamba for loss and error metrics.",
            "Threshold-dependent metrics use the fixed probability threshold of 0.5.",
            "Intervals quantify test-sample uncertainty, not training-seed variability.",
            "Weighted metrics use positive-label support as weights.",
        ],
    }

    for name in metric_names:
        a = observed_base.get(name)
        b = observed_mamba.get(name)
        difference = (b - a) if a is not None and b is not None else None

        output["models"]["resnet1d"][name] = {
            "observed": a,
            "ci95": interval(base_boot[name]),
        }
        output["models"]["ecg_mamba"][name] = {
            "observed": b,
            "ci95": interval(mamba_boot[name]),
        }
        output["differences_ecg_mamba_minus_resnet1d"][name] = {
            "observed": difference,
            "ci95": interval(diff_boot[name]),
        }

    RESULTS.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, allow_nan=False)

    print("\nPaired model comparison completed.")
    print(f"Test recordings: {len(patient_ids)}")
    print(f"Unique patients: {len(np.unique(patient_ids))}")
    print(f"Saved: {OUTPUT_PATH}")

    for name in (
        "auroc_macro", "auprc_macro", "f1_macro",
        "precision_macro", "recall_macro", "mcc_micro",
        "log_loss", "brier_macro", "ece_macro",
        "auroc_common", "auroc_medium", "auroc_rare",
        "auprc_common", "auprc_medium", "auprc_rare",
    ):
        if name not in output["differences_ecg_mamba_minus_resnet1d"]:
            continue
        result = output["differences_ecg_mamba_minus_resnet1d"][name]
        ci = result["ci95"]
        print(
            f"{name}: difference={result['observed']}, "
            f"95% CI=[{ci['lower']}, {ci['upper']}]"
        )


if __name__ == "__main__":
    main()
