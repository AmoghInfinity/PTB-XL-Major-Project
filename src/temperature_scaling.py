"""
Fit global temperature scaling on validation predictions and evaluate
calibration on validation and test splits.

The temperature is optimized using validation data only.
Test data is used exclusively for final evaluation.
Existing raw-calibration artifacts are not modified.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

VAL_PATH = RESULTS_DIR / "raw_calibration_validation_predictions.npz"
TEST_PATH = RESULTS_DIR / "raw_calibration_test_predictions.npz"

OUTPUT_JSON = RESULTS_DIR / "temperature_scaling_results.json"
OUTPUT_VAL = RESULTS_DIR / "temperature_scaled_validation_predictions.npz"
OUTPUT_TEST = RESULTS_DIR / "temperature_scaled_test_predictions.npz"


def binary_ece(y_true, y_prob, n_bins=15):
    """Calculate ECE for a binary label."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        if upper == 1.0:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)

        count = int(np.sum(mask))

        if count == 0:
            continue

        confidence = float(np.mean(y_prob[mask]))
        accuracy = float(np.mean(y_true[mask]))
        ece += (count / total) * abs(accuracy - confidence)

    return float(ece)


def per_label_ece(y_true, y_prob, n_bins=15):
    """Calculate ECE separately for every label."""
    return np.asarray(
        [
            binary_ece(y_true[:, i], y_prob[:, i], n_bins)
            for i in range(y_true.shape[1])
        ],
        dtype=np.float64,
    )


def per_label_brier(y_true, y_prob):
    """Calculate Brier score separately for every label."""
    return np.mean((y_prob - y_true) ** 2, axis=0)


def tier_summary(scores, tier_masks):
    """Calculate mean per-label scores for each frequency tier."""
    result = {}

    for tier, mask in tier_masks.items():
        result[tier] = (
            float(np.nanmean(scores[mask]))
            if np.any(mask)
            else float("nan")
        )

    return result


def load_predictions(path):
    """Load and validate a saved prediction artifact."""
    if not path.is_file():
        raise FileNotFoundError(f"Required predictions not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        y_true = np.asarray(data["y_true"], dtype=np.float64)
        y_prob = np.asarray(data["y_prob"], dtype=np.float64)
        vocab = np.asarray(data["vocab"]).astype(str)

    if y_true.shape != y_prob.shape:
        raise ValueError(f"Target and probability shapes differ in {path}")

    if y_true.ndim != 2 or y_true.shape[1] != len(vocab):
        raise ValueError(f"Prediction dimensions do not match vocabulary in {path}")

    if not np.isfinite(y_true).all() or not np.isfinite(y_prob).all():
        raise ValueError(f"Non-finite values found in {path}")

    if not np.isin(y_true, [0.0, 1.0]).all():
        raise ValueError(f"Targets must be binary in {path}")

    if np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError(f"Probabilities must be in [0, 1] in {path}")

    return y_true, y_prob, vocab


def load_tier_masks(vocab):
    """Load label-frequency tiers from the existing dataset definition."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data.dataset import PTBXLDataset

    dataset = PTBXLDataset(split="val")

    if not np.array_equal(
        np.asarray(dataset.vocab).astype(str),
        vocab,
    ):
        raise ValueError("Saved prediction vocabulary does not match dataset.")

    return {
        tier: np.asarray(dataset.tier_mask(tier), dtype=bool)
        for tier in ("common", "medium", "rare")
    }


def binary_cross_entropy_from_logits(y_true, logits):
    """Compute mean binary cross-entropy without numerical overflow."""
    return float(
        np.mean(
            np.maximum(logits, 0.0)
            - logits * y_true
            + np.log1p(np.exp(-np.abs(logits)))
        )
    )


def calculate_metrics(y_true, y_prob, tier_masks):
    """Calculate macro and frequency-tier ECE and Brier score."""
    ece_labels = per_label_ece(y_true, y_prob)
    brier_labels = per_label_brier(y_true, y_prob)

    return {
        "ece_macro": float(np.nanmean(ece_labels)),
        "ece_tiers": tier_summary(ece_labels, tier_masks),
        "brier_macro": float(np.nanmean(brier_labels)),
        "brier_tiers": tier_summary(brier_labels, tier_masks),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    y_val, p_val, vocab_val = load_predictions(VAL_PATH)
    y_test, p_test, vocab_test = load_predictions(TEST_PATH)

    if not np.array_equal(vocab_val, vocab_test):
        raise ValueError("Validation and test vocabularies do not match.")

    if y_val.shape[1] != y_test.shape[1]:
        raise ValueError("Validation and test label counts do not match.")

    tier_masks = load_tier_masks(vocab_val)

    # Convert saved probabilities to logits.
    # Clipping avoids infinite logits for probabilities equal to zero or one.
    epsilon = 1e-7
    logits_val = logit(np.clip(p_val, epsilon, 1.0 - epsilon))
    logits_test = logit(np.clip(p_test, epsilon, 1.0 - epsilon))

    # Fit one global temperature using validation binary cross-entropy.
    def validation_objective(log_temperature):
        temperature = np.exp(log_temperature)
        scaled_logits = logits_val / temperature
        return binary_cross_entropy_from_logits(
            y_val,
            scaled_logits,
        )

    optimization = minimize_scalar(
        validation_objective,
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
        options={"xatol": 1e-6},
    )

    if not optimization.success:
        raise RuntimeError(
            f"Temperature optimization failed: {optimization.message}"
        )

    temperature = float(np.exp(optimization.x))

    # Apply the fitted temperature to validation and test logits.
    p_val_scaled = expit(logits_val / temperature)
    p_test_scaled = expit(logits_test / temperature)

    results = {
        "method": "global_temperature_scaling",
        "temperature": temperature,
        "optimization": {
            "validation_bce_before": binary_cross_entropy_from_logits(
                y_val, logits_val
            ),
            "validation_bce_after": binary_cross_entropy_from_logits(
                y_val, logits_val / temperature
            ),
            "success": bool(optimization.success),
        },
        "validation": {
            "raw": calculate_metrics(y_val, p_val, tier_masks),
            "scaled": calculate_metrics(y_val, p_val_scaled, tier_masks),
        },
        "test": {
            "raw": calculate_metrics(y_test, p_test, tier_masks),
            "scaled": calculate_metrics(y_test, p_test_scaled, tier_masks),
        },
        "notes": [
            "Temperature was fitted on validation predictions only.",
            "Test predictions were not used for fitting.",
            (
                "Logits were reconstructed from saved probabilities; "
                "probabilities were clipped before logit conversion."
            ),
            (
                "ECE uses 15 equal-width bins and averages per-label ECE "
                "across the full label set or each frequency tier."
            ),
        ],
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, allow_nan=False)

    np.savez_compressed(
        OUTPUT_VAL,
        y_true=y_val,
        y_prob=p_val_scaled,
        vocab=vocab_val,
        temperature=np.asarray(temperature),
    )

    np.savez_compressed(
        OUTPUT_TEST,
        y_true=y_test,
        y_prob=p_test_scaled,
        vocab=vocab_test,
        temperature=np.asarray(temperature),
    )

    print("=" * 64)
    print("RESNET1D TEMPERATURE-SCALING RESULTS")
    print("=" * 64)
    print(f"Fitted temperature: {temperature:.6f}")
    print(
        "Validation BCE: "
        f"{results['optimization']['validation_bce_before']:.6f} -> "
        f"{results['optimization']['validation_bce_after']:.6f}"
    )

    for split in ("validation", "test"):
        print(f"\n{split.upper()}")

        for version in ("raw", "scaled"):
            metrics = results[split][version]
            print(f"\n  {version.capitalize()}")

            print(f"    Macro ECE   : {metrics['ece_macro']:.6f}")
            print(f"    Common ECE  : {metrics['ece_tiers']['common']:.6f}")
            print(f"    Medium ECE  : {metrics['ece_tiers']['medium']:.6f}")
            print(f"    Rare ECE    : {metrics['ece_tiers']['rare']:.6f}")

            print(f"    Macro Brier : {metrics['brier_macro']:.6f}")
            print(f"    Common Brier: {metrics['brier_tiers']['common']:.6f}")
            print(f"    Medium Brier: {metrics['brier_tiers']['medium']:.6f}")
            print(f"    Rare Brier  : {metrics['brier_tiers']['rare']:.6f}")

    print("\nSaved:")
    print(f"  {OUTPUT_JSON}")
    print(f"  {OUTPUT_VAL}")
    print(f"  {OUTPUT_TEST}")


if __name__ == "__main__":
    main()