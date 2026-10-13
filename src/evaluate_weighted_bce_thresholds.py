# importing libraries
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import PTBXLDataset
from models.resnet1d import ResNet1D


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_model_weighted_bce.pt"
)

VALIDATION_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "weighted_bce_validation_predictions.npz"
)

TEST_INPUT = (
    PROJECT_ROOT
    / "results"
    / "weighted_bce_test_predictions.npz"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "weighted_bce_threshold_evaluation.json"
)


def generate_validation_predictions(
    model,
    loader,
    device,
):
    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for signals, targets in loader:
            signals = signals.to(
                device,
                non_blocking=True,
            )

            logits = model(signals)

            probabilities = torch.sigmoid(
                logits
            ).cpu().numpy()

            all_probs.append(probabilities)
            all_targets.append(
                targets.numpy()
            )

    return (
        np.concatenate(all_targets),
        np.concatenate(all_probs),
    )


def find_best_thresholds(
    y_true,
    y_prob,
    threshold_grid,
):
    thresholds = np.zeros(
        y_true.shape[1],
        dtype=np.float32,
    )

    best_f1 = np.zeros(
        y_true.shape[1],
        dtype=np.float32,
    )

    for label_idx in range(y_true.shape[1]):
        for threshold in threshold_grid:
            predictions = (
                y_prob[:, label_idx]
                >= threshold
            ).astype(np.int32)

            score = f1_score(
                y_true[:, label_idx],
                predictions,
                zero_division=0,
            )

            if score > best_f1[label_idx]:
                best_f1[label_idx] = score
                thresholds[label_idx] = threshold

    return thresholds, best_f1


def get_tiers(vocab, validation_targets):
    counts = validation_targets.sum(axis=0)

    order = np.argsort(-counts)

    n_labels = len(vocab)
    third = n_labels // 3

    tier_of = {}

    for rank, label_idx in enumerate(order):
        if rank < third:
            tier = "common"
        elif rank < 2 * third:
            tier = "medium"
        else:
            tier = "rare"

        tier_of[vocab[label_idx]] = tier

    return tier_of


def evaluate_thresholds(
    y_true,
    y_prob,
    thresholds,
    vocab,
    tier_of,
):
    y_pred = (
        y_prob
        >= thresholds.reshape(1, -1)
    ).astype(np.int32)

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

    tier_results = {}

    for tier in [
        "common",
        "medium",
        "rare",
    ]:
        indices = [
            idx
            for idx, label in enumerate(vocab)
            if tier_of[label] == tier
        ]

        tier_results[tier] = {
            "f1": float(
                f1_score(
                    y_true[:, indices],
                    y_pred[:, indices],
                    average="macro",
                    zero_division=0,
                )
            ),
            "num_labels": len(indices),
        }

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "tier_results": tier_results,
        "positive_predictions": {
            label: int(
                y_pred[:, idx].sum()
            )
            for idx, label in enumerate(vocab)
        },
    }


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type == "cuda":
        print(
            f"Using GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print("Loading validation dataset...")

    validation_ds = PTBXLDataset(
        split="val"
    )

    validation_loader = DataLoader(
        validation_ds,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(
        f"Validation samples: "
        f"{len(validation_ds)}"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    checkpoint_vocab = np.asarray(
        checkpoint["vocab"]
    ).astype(str)

    dataset_vocab = np.asarray(
        validation_ds.vocab
    ).astype(str)

    if not np.array_equal(
        checkpoint_vocab,
        dataset_vocab,
    ):
        raise ValueError(
            "Checkpoint vocabulary does not "
            "match validation vocabulary."
        )

    model = ResNet1D(
        n_labels=len(validation_ds.vocab)
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    print(
        "\nGenerating weighted-BCE "
        "validation predictions..."
    )

    y_val_true, y_val_prob = (
        generate_validation_predictions(
            model,
            validation_loader,
            device,
        )
    )

    np.savez_compressed(
        VALIDATION_OUTPUT,
        y_true=y_val_true.astype(
            np.float32
        ),
        y_prob=y_val_prob.astype(
            np.float32
        ),
        vocab=np.asarray(
            validation_ds.vocab
        ),
    )

    print(
        f"Saved validation predictions to: "
        f"{VALIDATION_OUTPUT}"
    )

    threshold_grid = np.arange(
        0.01,
        1.0,
        0.01,
        dtype=np.float32,
    )

    print(
        "\nSelecting thresholds using "
        "validation data only..."
    )

    thresholds, validation_f1 = (
        find_best_thresholds(
            y_val_true,
            y_val_prob,
            threshold_grid,
        )
    )

    vocab = validation_ds.vocab

    tier_of = get_tiers(
        vocab,
        y_val_true,
    )

    print("\nSelected thresholds:")

    for label, threshold, score in zip(
        vocab,
        thresholds,
        validation_f1,
    ):
        print(
            f"{label:8s} "
            f"threshold={threshold:.2f} "
            f"validation_F1={score:.4f}"
        )

    print(
        "\nLoading existing weighted-BCE "
        "test predictions..."
    )

    test_data = np.load(
        TEST_INPUT
    )

    y_test_true = test_data[
        "y_true"
    ].astype(np.float32)

    y_test_prob = test_data[
        "y_prob"
    ].astype(np.float32)

    test_vocab = (
        test_data["vocab"]
        .astype(str)
    )

    if not np.array_equal(
        vocab,
        test_vocab,
    ):
        raise ValueError(
            "Validation and test "
            "vocabularies do not match."
        )

    print(
        "\nEvaluating frozen validation "
        "thresholds on test data..."
    )

    test_results = evaluate_thresholds(
        y_test_true,
        y_test_prob,
        thresholds,
        vocab,
        tier_of,
    )

    print(
        "\nWeighted BCE threshold-selected "
        "test results"
    )
    print(
        "======================================"
    )

    print(
        f"Macro F1: "
        f"{test_results['macro_f1']:.4f}"
    )

    print(
        f"Micro F1: "
        f"{test_results['micro_f1']:.4f}"
    )

    for tier, result in (
        test_results["tier_results"].items()
    ):
        print(
            f"{tier.capitalize():8s} "
            f"F1={result['f1']:.4f}"
        )

    results = {
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "validation_samples": int(
            len(y_val_true)
        ),
        "test_samples": int(
            len(y_test_true)
        ),
        "threshold_selection": {
            "method": (
                "per-label F1 maximization "
                "on validation set"
            ),
            "threshold_min": 0.01,
            "threshold_max": 0.99,
            "threshold_step": 0.01,
        },
        "thresholds": {
            label: float(threshold)
            for label, threshold in zip(
                vocab,
                thresholds,
            )
        },
        "validation_best_f1": {
            label: float(score)
            for label, score in zip(
                vocab,
                validation_f1,
            )
        },
        "test_results": test_results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print(
        f"\nSaved results to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()