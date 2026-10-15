
# importing libraries
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

VAL_PATH = RESULTS_DIR / "ecg_mamba_validation_predictions.npz"
TEST_PATH = RESULTS_DIR / "ecg_mamba_test_predictions.npz"

OUTPUT_JSON = RESULTS_DIR / "ecg_mamba_temperature_scaling_results.json"
OUTPUT_VAL = RESULTS_DIR / "ecg_mamba_temperature_scaled_validation_predictions.npz"
OUTPUT_TEST = RESULTS_DIR / "ecg_mamba_temperature_scaled_test_predictions.npz"


def load_predictions(path):
    if not path.is_file():
        raise FileNotFoundError(f"Predictions not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        y_true = np.asarray(data["y_true"], dtype=np.float64)
        y_prob = np.asarray(data["y_prob"], dtype=np.float64)
        vocab = np.asarray(data["vocab"]).astype(str)

    if y_true.shape != y_prob.shape:
        raise ValueError(f"Target/probability shape mismatch in {path}")
    if y_true.ndim != 2 or y_true.shape[1] != len(vocab):
        raise ValueError(f"Vocabulary shape mismatch in {path}")
    if not np.isfinite(y_true).all() or not np.isfinite(y_prob).all():
        raise ValueError(f"Non-finite values in {path}")
    if not np.isin(y_true, [0.0, 1.0]).all():
        raise ValueError(f"Targets must be binary in {path}")
    if np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError(f"Probabilities outside [0, 1] in {path}")

    return y_true, y_prob, vocab


def binary_ece(y_true, y_prob, n_bins=15):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)

        if mask.any():
            ece += mask.mean() * abs(
                y_true[mask].mean() - y_prob[mask].mean()
            )

    return float(ece)


def metrics(y_true, y_prob):
    per_label_ece = [
        binary_ece(y_true[:, j], y_prob[:, j])
        for j in range(y_true.shape[1])
    ]
    per_label_brier = np.mean((y_prob - y_true) ** 2, axis=0)

    return {
        "macro_ece": float(np.mean(per_label_ece)),
        "macro_brier": float(np.mean(per_label_brier)),
    }


def bce_from_logits(y_true, logits):
    return float(np.mean(
        np.maximum(logits, 0.0)
        - logits * y_true
        + np.log1p(np.exp(-np.abs(logits)))
    ))


def main():
    y_val, p_val, vocab_val = load_predictions(VAL_PATH)
    y_test, p_test, vocab_test = load_predictions(TEST_PATH)

    if not np.array_equal(vocab_val, vocab_test):
        raise ValueError("Validation and test vocabularies differ.")

    if not np.array_equal(y_test.shape[1:], y_val.shape[1:]):
        raise ValueError("Validation and test label dimensions differ.")

    epsilon = 1e-7
    logits_val = logit(np.clip(p_val, epsilon, 1.0 - epsilon))
    logits_test = logit(np.clip(p_test, epsilon, 1.0 - epsilon))

    def objective(log_temperature):
        temperature = np.exp(log_temperature)
        return bce_from_logits(y_val, logits_val / temperature)

    optimization = minimize_scalar(
        objective,
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
        options={"xatol": 1e-6},
    )

    if not optimization.success:
        raise RuntimeError(f"Temperature fitting failed: {optimization.message}")

    temperature = float(np.exp(optimization.x))
    p_val_scaled = expit(logits_val / temperature)
    p_test_scaled = expit(logits_test / temperature)

    results = {
        "model": "ecg_mamba",
        "method": "global_temperature_scaling",
        "temperature": temperature,
        "validation_bce_before": bce_from_logits(y_val, logits_val),
        "validation_bce_after": bce_from_logits(
            y_val, logits_val / temperature
        ),
        "validation": {
            "raw": metrics(y_val, p_val),
            "scaled": metrics(y_val, p_val_scaled),
        },
        "test": {
            "raw": metrics(y_test, p_test),
            "scaled": metrics(y_test, p_test_scaled),
        },
        "notes": [
            "Temperature was fitted using validation predictions only.",
            "Test predictions were used only for final evaluation.",
            "ECE uses 15 equal-width bins and averages per-label ECE.",
            "Logits were reconstructed from saved probabilities after clipping.",
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

    print("=" * 56)
    print("ECG-MAMBA TEMPERATURE SCALING")
    print("=" * 56)
    print(f"Temperature: {temperature:.6f}")
    print(
        "Validation BCE: "
        f"{results['validation_bce_before']:.6f} -> "
        f"{results['validation_bce_after']:.6f}"
    )

    for split in ("validation", "test"):
        print(f"\n{split.upper()}")
        for version in ("raw", "scaled"):
            item = results[split][version]
            print(
                f"  {version:6s} | "
                f"ECE={item['macro_ece']:.6f} | "
                f"Brier={item['macro_brier']:.6f}"
            )

    print("\nSaved:")
    print(OUTPUT_JSON)
    print(OUTPUT_VAL)
    print(OUTPUT_TEST)


if __name__ == "__main__":
    main()
