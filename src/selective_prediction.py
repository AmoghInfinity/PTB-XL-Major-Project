from pathlib import Path
import csv
import json

import numpy as np
from sklearn.metrics import f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

MODELS = {
    "resnet1d": RESULTS / "baseline_test_predictions.npz",
    "ecg_mamba": RESULTS / "ecg_mamba_test_predictions.npz",
}

THRESHOLD = 0.5
COVERAGE_LEVELS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)


def load_predictions(path):
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        y_true = data["y_true"].astype(np.int32)
        y_prob = data["y_prob"].astype(np.float64)

    if y_true.shape != y_prob.shape:
        raise ValueError(f"Target and probability shapes differ: {path}")

    if y_true.ndim != 2:
        raise ValueError(f"Expected a 2D multi-label array: {path}")

    if not np.all(np.isfinite(y_prob)):
        raise ValueError(f"Non-finite probabilities found: {path}")

    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError(f"Probabilities outside [0, 1]: {path}")

    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError(f"Targets must be binary: {path}")

    return y_true, y_prob


def evaluate_model(y_true, y_prob):
    predictions = (y_prob >= THRESHOLD).astype(np.int32)
    n_records, n_labels = y_true.shape

    eps = np.finfo(np.float64).eps
    probabilities = np.clip(y_prob, eps, 1.0 - eps)

    entropy = -(
        probabilities * np.log(probabilities)
        + (1.0 - probabilities) * np.log(1.0 - probabilities)
    ) / np.log(2.0)

    record_uncertainty = np.mean(entropy, axis=1)
    record_risk = np.mean(predictions != y_true, axis=1)

    order = np.argsort(record_uncertainty, kind="stable")
    sorted_risk = record_risk[order]

    accepted_counts = np.arange(1, n_records + 1)
    cumulative_risk = np.cumsum(sorted_risk) / accepted_counts
    coverage_curve = accepted_counts / n_records

    full_coverage = np.concatenate(([0.0], coverage_curve))
    full_risk = np.concatenate(([0.0], cumulative_risk))

    aurc = float(np.sum(
        np.diff(full_coverage)
        * (full_risk[:-1] + full_risk[1:])
        / 2.0
    ))

    points = []

    for requested_coverage in COVERAGE_LEVELS:
        count = int(np.ceil(requested_coverage * n_records))
        selected = order[:count]

        y_selected = y_true[selected]
        p_selected = predictions[selected]

        points.append({
            "requested_coverage": requested_coverage,
            "actual_coverage": count / n_records,
            "accepted_records": count,
            "mean_uncertainty": float(
                np.mean(record_uncertainty[selected])
            ),
            "hamming_risk": float(cumulative_risk[count - 1]),
            "macro_f1": float(f1_score(
                y_selected, p_selected,
                average="macro", zero_division=0
            )),
            "micro_f1": float(f1_score(
                y_selected, p_selected,
                average="micro", zero_division=0
            )),
            "macro_recall": float(recall_score(
                y_selected, p_selected,
                average="macro", zero_division=0
            )),
        })

    curve = [
        {
            "accepted_records": int(k),
            "coverage": float(c),
            "hamming_risk": float(r),
        }
        for k, c, r in zip(
            accepted_counts, coverage_curve, cumulative_risk
        )
    ]

    return {
        "recordings": int(n_records),
        "labels": int(n_labels),
        "threshold": THRESHOLD,
        "uncertainty_method": "mean_normalized_binary_entropy",
        "risk_definition": "mean_per_record_hamming_loss",
        "aurc": aurc,
        "coverage_points": points,
        "curve": curve,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    output = {
        "method": "record_level_selective_prediction",
        "selection_rule": (
            "Accept recordings in ascending order of mean normalized "
            "binary entropy across labels."
        ),
        "models": {},
    }

    csv_path = RESULTS / "selective_prediction_curve.csv"
    json_path = RESULTS / "selective_prediction_results.json"

    reference_targets = None

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "model", "accepted_records", "coverage", "hamming_risk"
            ],
        )
        writer.writeheader()

        for model_name, path in MODELS.items():
            y_true, y_prob = load_predictions(path)

            if reference_targets is None:
                reference_targets = y_true
            elif not np.array_equal(reference_targets, y_true):
                raise ValueError(
                    "Model test targets differ; paired comparison is invalid."
                )

            result = evaluate_model(y_true, y_prob)
            output["models"][model_name] = {
                key: value for key, value in result.items()
                if key != "curve"
            }

            for row in result["curve"]:
                writer.writerow({"model": model_name, **row})

            print(
                f"{model_name}: recordings={result['recordings']}, "
                f"labels={result['labels']}, AURC={result['aurc']:.6f}"
            )

            for point in result["coverage_points"]:
                print(
                    f"  coverage={point['actual_coverage']:.3f}, "
                    f"risk={point['hamming_risk']:.6f}, "
                    f"macro_F1={point['macro_f1']:.6f}, "
                    f"micro_F1={point['micro_f1']:.6f}"
                )

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(output, json_file, indent=2, allow_nan=False)

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
