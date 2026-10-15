"""
Patient-level bootstrap analysis for the frozen PTB-XL ECG-Mamba model.

Compares raw and temperature-scaled predictions on the official test fold.
Patients are resampled as clusters to account for multiple recordings
belonging to the same patient.

All outputs are written to new files.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

RAW_PATH = RESULTS_DIR / "ecg_mamba_test_predictions.npz"
SCALED_PATH = RESULTS_DIR / "ecg_mamba_temperature_scaled_test_predictions.npz"
OUTPUT_PATH = RESULTS_DIR / "ecg_mamba_statistical_analysis_results.json"

N_BOOTSTRAPS = 1000
RANDOM_SEED = 42
F1_THRESHOLD = 0.5
ECE_BINS = 15


def load_predictions(path):
    """Load saved predictions and validate their contents."""
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        y_true = np.asarray(data["y_true"], dtype=np.float64)
        y_prob = np.asarray(data["y_prob"], dtype=np.float64)
        vocab = np.asarray(data["vocab"]).astype(str)

    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch in {path}")

    if y_true.ndim != 2 or y_true.shape[1] != len(vocab):
        raise ValueError(f"Vocabulary mismatch in {path}")

    if not np.isfinite(y_true).all() or not np.isfinite(y_prob).all():
        raise ValueError(f"Non-finite values found in {path}")

    if not np.isin(y_true, [0.0, 1.0]).all():
        raise ValueError(f"Targets must be binary in {path}")

    if np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError(f"Probabilities outside [0, 1] in {path}")

    return y_true, y_prob, vocab


def get_patient_groups():
    """Return a patient identifier for every test recording."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data.dataset import PTBXLDataset

    dataset = PTBXLDataset(split="test")

    if "patient_id" not in dataset.df.columns:
        raise ValueError("Test metadata does not contain patient_id.")

    patient_ids = dataset.df["patient_id"].to_numpy()

    if len(patient_ids) != 2198:
        raise ValueError(
            f"Expected 2198 test records, found {len(patient_ids)}."
        )

    return patient_ids, dataset


def make_tier_masks(dataset, vocab):
    """Construct frequency-tier masks using the dataset definition."""
    dataset_vocab = np.asarray(dataset.vocab).astype(str)

    if not np.array_equal(dataset_vocab, vocab):
        raise ValueError("Prediction vocabulary does not match test dataset.")

    return {
        tier: np.asarray(dataset.tier_mask(tier), dtype=bool)
        for tier in ("common", "medium", "rare")
    }


def binary_ece(y_true, y_prob, n_bins=ECE_BINS):
    """Compute expected calibration error for one binary label."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        lower = edges[i]
        upper = edges[i + 1]

        if i == n_bins - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)

        count = int(mask.sum())

        if count == 0:
            continue

        confidence = float(y_prob[mask].mean())
        frequency = float(y_true[mask].mean())

        ece += (count / total) * abs(frequency - confidence)

    return float(ece)


def calculate_metrics(y_true, y_prob, tier_masks):
    """Calculate macro and frequency-tier metrics."""
    auroc = []
    auprc = []
    f1_scores = []
    ece_scores = []
    brier_scores = []

    y_pred = (y_prob >= F1_THRESHOLD).astype(np.int32)

    for i in range(y_true.shape[1]):
        target = y_true[:, i]
        probability = y_prob[:, i]

        if np.unique(target).size < 2:
            auroc.append(np.nan)
            auprc.append(np.nan)
        else:
            auroc.append(roc_auc_score(target, probability))
            auprc.append(average_precision_score(target, probability))

        f1_scores.append(
            f1_score(
                target,
                y_pred[:, i],
                zero_division=0,
            )
        )

        ece_scores.append(binary_ece(target, probability))
        brier_scores.append(np.mean((target - probability) ** 2))

    scores = {
        "auroc": np.asarray(auroc, dtype=np.float64),
        "auprc": np.asarray(auprc, dtype=np.float64),
        "f1": np.asarray(f1_scores, dtype=np.float64),
        "ece": np.asarray(ece_scores, dtype=np.float64),
        "brier": np.asarray(brier_scores, dtype=np.float64),
    }

    summary = {}

    for metric, values in scores.items():
        summary[f"{metric}_macro"] = float(np.nanmean(values))

        for tier, mask in tier_masks.items():
            summary[f"{metric}_{tier}"] = (
                float(np.nanmean(values[mask]))
                if np.any(mask)
                else float("nan")
            )

    return summary


def bootstrap_indices(patient_ids, rng):
    """Resample patients with replacement and retain all their records."""
    unique_patients = np.unique(patient_ids)

    sampled_patients = rng.choice(
        unique_patients,
        size=len(unique_patients),
        replace=True,
    )

    sampled_indices = []

    for patient_id in sampled_patients:
        indices = np.flatnonzero(patient_ids == patient_id)
        sampled_indices.append(indices)

    return np.concatenate(sampled_indices)


def percentile_interval(values):
    """Return a percentile 95% confidence interval."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "lower": None,
            "upper": None,
            "valid_bootstraps": 0,
        }

    lower, upper = np.percentile(values, [2.5, 97.5])

    return {
        "lower": float(lower),
        "upper": float(upper),
        "valid_bootstraps": int(len(values)),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    y_raw, p_raw, vocab_raw = load_predictions(RAW_PATH)
    y_scaled, p_scaled, vocab_scaled = load_predictions(SCALED_PATH)

    if not np.array_equal(vocab_raw, vocab_scaled):
        raise ValueError("Raw and scaled vocabularies do not match.")

    if y_raw.shape != y_scaled.shape:
        raise ValueError("Raw and scaled prediction shapes do not match.")

    if not np.array_equal(y_raw, y_scaled):
        raise ValueError("Raw and scaled targets do not match.")

    patient_ids, dataset = get_patient_groups()

    if len(patient_ids) != len(y_raw):
        raise ValueError("Patient IDs do not align with prediction records.")

    tier_masks = make_tier_masks(dataset, vocab_raw)

    # Calculate observed metrics on the complete test fold.
    raw_observed = calculate_metrics(y_raw, p_raw, tier_masks)
    scaled_observed = calculate_metrics(y_scaled, p_scaled, tier_masks)

    # Resample the same patients for both models to obtain paired differences.
    rng = np.random.default_rng(RANDOM_SEED)

    metric_names = list(raw_observed.keys())
    raw_bootstrap = {name: [] for name in metric_names}
    scaled_bootstrap = {name: [] for name in metric_names}
    difference_bootstrap = {name: [] for name in metric_names}

    for iteration in range(N_BOOTSTRAPS):
        indices = bootstrap_indices(patient_ids, rng)

        y_sample = y_raw[indices]
        raw_sample = p_raw[indices]
        scaled_sample = p_scaled[indices]

        raw_metrics = calculate_metrics(
            y_sample,
            raw_sample,
            tier_masks,
        )
        scaled_metrics = calculate_metrics(
            y_sample,
            scaled_sample,
            tier_masks,
        )

        for name in metric_names:
            raw_value = raw_metrics[name]
            scaled_value = scaled_metrics[name]

            raw_bootstrap[name].append(raw_value)
            scaled_bootstrap[name].append(scaled_value)
            difference_bootstrap[name].append(
                scaled_value - raw_value
            )

        if (iteration + 1) % 100 == 0:
            print(
                f"Completed {iteration + 1}/{N_BOOTSTRAPS} bootstraps"
            )

    output = {
        "method": "patient_level_paired_bootstrap",
        "n_bootstraps": N_BOOTSTRAPS,
        "random_seed": RANDOM_SEED,
        "f1_threshold": F1_THRESHOLD,
        "ece_bins": ECE_BINS,
        "test_recordings": int(len(patient_ids)),
        "unique_patients": int(len(np.unique(patient_ids))),
        "temperature": float(
            np.load(SCALED_PATH, allow_pickle=False)["temperature"]
        ),
        "metrics": {},
        "interpretation_notes": [
            (
                "The same patient bootstrap samples are used for raw and "
                "temperature-scaled predictions."
            ),
            (
                "Differences are scaled minus raw. Negative differences "
                "favor temperature scaling for ECE and Brier score; positive differences favor temperature scaling for AUROC, AUPRC, and F1. A zero difference indicates no observed change."
            ),
            (
                "AUROC and AUPRC bootstrap replicates with undefined "
                "per-label values are summarized using nanmean."
            ),
            (
                "The intervals quantify test-sample uncertainty under "
                "patient-cluster resampling, not training-seed variability."
            ),
        ],
    }

    for name in metric_names:
        output["metrics"][name] = {
            "raw_observed": raw_observed[name],
            "raw_ci95": percentile_interval(raw_bootstrap[name]),
            "scaled_observed": scaled_observed[name],
            "scaled_ci95": percentile_interval(scaled_bootstrap[name]),
            "difference_scaled_minus_raw": (
                scaled_observed[name] - raw_observed[name]
            ),
            "difference_ci95": percentile_interval(
                difference_bootstrap[name]
            ),
        }

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, allow_nan=False)

    print("\n" + "=" * 68)
    print("ECG-MAMBA PATIENT-LEVEL STATISTICAL ANALYSIS")
    print("=" * 68)
    print(f"Test recordings: {len(patient_ids)}")
    print(f"Unique patients: {len(np.unique(patient_ids))}")
    print(f"Bootstrap replicates: {N_BOOTSTRAPS}")

    for name in metric_names:
        result = output["metrics"][name]
        raw_value = result["raw_observed"]
        scaled_value = result["scaled_observed"]
        difference = result["difference_scaled_minus_raw"]
        ci = result["difference_ci95"]

        print(f"\n{name}")
        print(f"  Raw:      {raw_value:.6f}")
        print(f"  Scaled:   {scaled_value:.6f}")
        print(f"  Difference (scaled - raw): {difference:.6f}")

        if ci["lower"] is not None:
            print(
                f"  Difference 95% CI: "
                f"[{ci['lower']:.6f}, {ci['upper']:.6f}]"
            )

    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
