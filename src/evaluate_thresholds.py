# importing libraries
import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import PTBXLDataset
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
)


def load_predictions(path):
    data = np.load(path)

    return (
        data["y_true"].astype(np.float32),
        data["y_prob"].astype(np.float32),
        data["vocab"].astype(str),
    )


def find_best_thresholds(y_true, y_prob, threshold_grid):
    thresholds = np.zeros(y_true.shape[1], dtype=np.float32)
    best_f1 = np.zeros(y_true.shape[1], dtype=np.float32)

    for label_idx in range(y_true.shape[1]):
        label_true = y_true[:, label_idx]
        label_prob = y_prob[:, label_idx]

        for threshold in threshold_grid:
            predictions = (label_prob >= threshold).astype(np.int32)
            score = f1_score(
                label_true,
                predictions,
                zero_division=0,
            )

            if score > best_f1[label_idx]:
                best_f1[label_idx] = score
                thresholds[label_idx] = threshold

    return thresholds, best_f1


def evaluate_predictions(y_true, y_prob, thresholds, vocab, tier_of):
    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(np.int32)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    micro_f1 = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    macro_auroc = roc_auc_score(
        y_true,
        y_prob,
        average="macro",
    )

    tier_results = {}

    for tier in ["common", "medium", "rare"]:
        indices = [
            idx
            for idx, label in enumerate(vocab)
            if tier_of.get(label) == tier
        ]

        if not indices:
            continue

        tier_results[tier] = {
            "f1": float(
                f1_score(
                    y_true[:, indices],
                    y_pred[:, indices],
                    average="macro",
                    zero_division=0,
                )
            ),
            "auroc": float(
                roc_auc_score(
                    y_true[:, indices],
                    y_prob[:, indices],
                    average="macro",
                )
            ),
            "num_labels": len(indices),
        }

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "macro_auroc": float(macro_auroc),
        "tier_results": tier_results,
        "thresholds": {
            label: float(threshold)
            for label, threshold in zip(vocab, thresholds)
        },
        "positive_predictions": {
            label: int(y_pred[:, idx].sum())
            for idx, label in enumerate(vocab)
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Select per-label thresholds on validation data and evaluate on test data."
    )

    parser.add_argument(
        "--validation",
        type=str,
        default="results/raw_calibration_validation_predictions.npz",
    )

    parser.add_argument(
        "--test",
        type=str,
        default="results/baseline_test_predictions.npz",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/threshold_evaluation_tiers_corrected.json",
    )

    parser.add_argument(
        "--step",
        type=float,
        default=0.01,
        help="Threshold search step.",
    )

    args = parser.parse_args()

    validation_path = Path(args.validation)
    test_path = Path(args.test)
    output_path = Path(args.output)

    print("Loading validation predictions...")
    val_y_true, val_y_prob, val_vocab = load_predictions(validation_path)

    print("Loading test predictions...")
    test_y_true, test_y_prob, test_vocab = load_predictions(test_path)

    if not np.array_equal(val_vocab, test_vocab):
        raise ValueError("Validation and test vocabularies do not match.")

    vocab = val_vocab

    threshold_grid = np.arange(
        args.step,
        1.0,
        args.step,
        dtype=np.float32,
    )

    print(f"Validation samples: {len(val_y_true)}")
    print(f"Test samples: {len(test_y_true)}")
    print(f"Labels: {len(vocab)}")
    print(f"Thresholds searched: {len(threshold_grid)}")

    print("\nSelecting thresholds using validation data only...")

    thresholds, validation_best_f1 = find_best_thresholds(
        val_y_true,
        val_y_prob,
        threshold_grid,
    )

    tier_ds = PTBXLDataset(split="val")

    if not np.array_equal(
        np.asarray(vocab, dtype=str),
        np.asarray(tier_ds.vocab, dtype=str),
    ):
        raise ValueError(
            "Prediction vocabulary does not match the dataset vocabulary."
        )

    tier_of = tier_ds.tier_of

    print("\nValidation-selected thresholds:")
    for label, threshold, score in zip(
        vocab,
        thresholds,
        validation_best_f1,
    ):
        print(
            f"{label:8s} "
            f"threshold={threshold:.2f} "
            f"validation_F1={score:.4f}"
        )

    print("\nEvaluating validation-selected thresholds on test data...")

    threshold_results = evaluate_predictions(
        test_y_true,
        test_y_prob,
        thresholds,
        vocab,
        tier_of,
    )

    print("\nThreshold-selected test results")
    print("--------------------------------")
    print(f"Macro F1:    {threshold_results['macro_f1']:.4f}")
    print(f"Micro F1:    {threshold_results['micro_f1']:.4f}")
    print(f"Macro AUROC: {threshold_results['macro_auroc']:.4f}")

    for tier, result in threshold_results["tier_results"].items():
        print(
            f"{tier.capitalize():8s} "
            f"F1={result['f1']:.4f} "
            f"AUROC={result['auroc']:.4f}"
        )

    output = {
        "validation_source": str(validation_path),
        "test_source": str(test_path),
        "threshold_selection": {
            "method": "per-label F1 maximization on validation set",
            "threshold_min": float(args.step),
            "threshold_max": float(1.0 - args.step),
            "threshold_step": float(args.step),
        },
        "results": threshold_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()