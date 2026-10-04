"""
Raw calibration analysis for the PTB-XL baseline.

This script evaluates the frozen baseline model without changing its
probabilities. It computes:
- Expected Calibration Error (ECE)
- Brier score
- Per-frequency-tier ECE/Brier
- Reliability diagrams

Validation results are used as the calibration reference set.
Test results are reported separately and are not used to fit anything.

The saved outputs can later be reused for temperature scaling.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import PTBXLDataset
from models.resnet1d import ResNet1D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "calibration"


def get_device():
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")

    print("WARNING: CUDA not available. Using CPU.")
    return torch.device("cpu")


def binary_ece(y_true, y_prob, n_bins=15):
    """
    Calculate Expected Calibration Error for one binary label.

    ECE is the weighted absolute difference between empirical accuracy
    and mean confidence across probability bins.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        if upper == 1.0:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)

        count = np.sum(mask)

        if count == 0:
            continue

        confidence = np.mean(y_prob[mask])
        accuracy = np.mean(y_true[mask])

        ece += (count / total) * abs(accuracy - confidence)

    return float(ece)


def per_label_ece(y_true, y_prob, n_bins=15):
    """Calculate ECE independently for every label."""
    scores = []

    for i in range(y_true.shape[1]):
        scores.append(
            binary_ece(
                y_true[:, i],
                y_prob[:, i],
                n_bins=n_bins,
            )
        )

    return np.asarray(scores, dtype=np.float64)


def per_label_brier(y_true, y_prob):
    """Calculate Brier score independently for every label."""
    return np.mean(
        (y_prob - y_true) ** 2,
        axis=0,
    )


def tier_summary(scores, dataset):
    """Average per-label scores within each frequency tier."""
    results = {}

    for tier in ["common", "medium", "rare"]:
        mask = dataset.tier_mask(tier)

        if np.any(mask):
            results[tier] = float(np.mean(scores[mask]))
        else:
            results[tier] = float("nan")

    return results


def collect_predictions(model, loader, device):
    """Run inference and return targets and sigmoid probabilities."""
    model.eval()

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for signals, targets in loader:
            signals = signals.to(
                device,
                non_blocking=True,
            )

            logits = model(signals)
            probs = torch.sigmoid(logits)

            all_targets.append(
                targets.cpu().numpy()
            )
            all_probs.append(
                probs.cpu().numpy()
            )

    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    return y_true, y_prob


def load_model(device, n_labels):
    """Load the frozen best baseline checkpoint."""
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model = ResNet1D(
        n_labels=n_labels,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    return model


def reliability_data(y_true, y_prob, n_bins=15):
    """Build bin-level reliability statistics."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    mean_confidence = []
    mean_accuracy = []
    counts = []

    for lower, upper in zip(
        bin_edges[:-1],
        bin_edges[1:],
    ):
        if upper == 1.0:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)

        count = np.sum(mask)

        if count == 0:
            continue

        mean_confidence.append(
            float(np.mean(y_prob[mask]))
        )

        mean_accuracy.append(
            float(np.mean(y_true[mask]))
        )

        counts.append(int(count))

    return (
        np.asarray(mean_confidence),
        np.asarray(mean_accuracy),
        np.asarray(counts),
    )


def save_reliability_diagram(
    y_true,
    y_prob,
    title,
    output_path,
    n_bins=15,
):
    """
    Save a reliability diagram.

    The diagram pools label-level predictions within the requested
    frequency tier. Numerical ECE is still calculated per label and
    averaged across labels, which avoids a single high-frequency label
    dominating the reported metric.
    """
    confidence, accuracy, counts = reliability_data(
        y_true,
        y_prob,
        n_bins=n_bins,
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        label="Perfect calibration",
    )

    if len(confidence) > 0:
        plt.plot(
            confidence,
            accuracy,
            marker="o",
            label="Model",
        )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical positive frequency")
    plt.title(title)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
    )
    plt.close()


def evaluate_split(
    split,
    model,
    batch_size,
    num_workers,
    device,
    n_bins,
):
    """Evaluate raw calibration on one split."""
    dataset = PTBXLDataset(split=split)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(
        f"\nLoading {split} split: "
        f"{len(dataset)} recordings, "
        f"{len(dataset.vocab)} labels"
    )

    y_true, y_prob = collect_predictions(
        model,
        loader,
        device,
    )

    ece_labels = per_label_ece(
        y_true,
        y_prob,
        n_bins=n_bins,
    )

    brier_labels = per_label_brier(
        y_true,
        y_prob,
    )

    ece_tiers = tier_summary(
        ece_labels,
        dataset,
    )

    brier_tiers = tier_summary(
        brier_labels,
        dataset,
    )

    return {
        "dataset": dataset,
        "y_true": y_true,
        "y_prob": y_prob,
        "ece_per_label": ece_labels,
        "brier_per_label": brier_labels,
        "ece_macro": float(np.mean(ece_labels)),
        "brier_macro": float(np.mean(brier_labels)),
        "ece_tiers": ece_tiers,
        "brier_tiers": brier_tiers,
    }


def print_results(split, results):
    print("\n" + "=" * 60)
    print(f"RAW CALIBRATION — {split.upper()}")
    print("=" * 60)

    print("\nECE")
    print(f"  Macro : {results['ece_macro']:.6f}")

    for tier in ["common", "medium", "rare"]:
        print(
            f"  {tier.capitalize():7s}: "
            f"{results['ece_tiers'][tier]:.6f}"
        )

    print("\nBrier Score")
    print(f"  Macro : {results['brier_macro']:.6f}")

    for tier in ["common", "medium", "rare"]:
        print(
            f"  {tier.capitalize():7s}: "
            f"{results['brier_tiers'][tier]:.6f}"
        )

    print("=" * 60)


def save_results(val_results, test_results):
    """Save numerical calibration results for later experiments."""
    output = {
        "validation": {
            "ece_macro": val_results["ece_macro"],
            "ece_common": val_results["ece_tiers"]["common"],
            "ece_medium": val_results["ece_tiers"]["medium"],
            "ece_rare": val_results["ece_tiers"]["rare"],
            "brier_macro": val_results["brier_macro"],
            "brier_common": val_results["brier_tiers"]["common"],
            "brier_medium": val_results["brier_tiers"]["medium"],
            "brier_rare": val_results["brier_tiers"]["rare"],
        },
        "test": {
            "ece_macro": test_results["ece_macro"],
            "ece_common": test_results["ece_tiers"]["common"],
            "ece_medium": test_results["ece_tiers"]["medium"],
            "ece_rare": test_results["ece_tiers"]["rare"],
            "brier_macro": test_results["brier_macro"],
            "brier_common": test_results["brier_tiers"]["common"],
            "brier_medium": test_results["brier_tiers"]["medium"],
            "brier_rare": test_results["brier_tiers"]["rare"],
        },
    }

    import json

    with open(
        RESULTS_DIR / "raw_calibration_results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=15,
    )

    args = parser.parse_args()

    RESULTS_DIR.mkdir(
        exist_ok=True
    )

    PLOTS_DIR.mkdir(
        exist_ok=True
    )

    device = get_device()

    print(
        f"Loading checkpoint: {CHECKPOINT_PATH}"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    n_labels = len(
        checkpoint["vocab"]
    )

    model = load_model(
        device,
        n_labels,
    )

    print(
        "Checkpoint loaded successfully."
    )

    # Validation is the calibration reference split.
    val_results = evaluate_split(
        split="val",
        model=model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        n_bins=args.bins,
    )

    # Test is reported only as final raw baseline performance.
    test_results = evaluate_split(
        split="test",
        model=model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        n_bins=args.bins,
    )

    print_results(
        "validation",
        val_results,
    )

    print_results(
        "test",
        test_results,
    )

    # Build tier-specific reliability diagrams.
    for tier in ["common", "medium", "rare"]:
        val_mask = val_results["dataset"].tier_mask(
            tier
        )

        save_reliability_diagram(
            val_results["y_true"][:, val_mask].reshape(-1),
            val_results["y_prob"][:, val_mask].reshape(-1),
            f"Raw Calibration — Validation — {tier.capitalize()}",
            PLOTS_DIR / f"reliability_val_{tier}.png",
            n_bins=args.bins,
        )

        test_mask = test_results["dataset"].tier_mask(
            tier
        )

        save_reliability_diagram(
            test_results["y_true"][:, test_mask].reshape(-1),
            test_results["y_prob"][:, test_mask].reshape(-1),
            f"Raw Calibration — Test — {tier.capitalize()}",
            PLOTS_DIR / f"reliability_test_{tier}.png",
            n_bins=args.bins,
        )

    # Save raw predictions so the next calibration stage can reuse them.
    np.savez_compressed(
        RESULTS_DIR / "raw_calibration_validation_predictions.npz",
        y_true=val_results["y_true"],
        y_prob=val_results["y_prob"],
        vocab=np.asarray(
            val_results["dataset"].vocab
        ),
    )

    np.savez_compressed(
        RESULTS_DIR / "raw_calibration_test_predictions.npz",
        y_true=test_results["y_true"],
        y_prob=test_results["y_prob"],
        vocab=np.asarray(
            test_results["dataset"].vocab
        ),
    )

    save_results(
        val_results,
        test_results,
    )

    print(
        "\nSaved calibration artifacts to:"
    )
    print(
        f"  {RESULTS_DIR}"
    )
    print(
        f"  {PLOTS_DIR}"
    )


if __name__ == "__main__":
    main()
