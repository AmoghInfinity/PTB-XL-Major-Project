"""
Evaluate a PTB-XL model on the official test fold.

Reports:
- Macro AUROC
- Per-tier AUROC
- Macro F1
- Per-tier F1

Saves raw probabilities and targets for downstream
calibration and selective-prediction experiments.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import PTBXLDataset
from models.resnet1d import ResNet1D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device

    print("WARNING: CUDA not available. Using CPU.")
    return torch.device("cpu")


def per_label_auroc(y_true, y_prob):
    """Calculate AUROC for each label."""
    scores = []

    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            scores.append(np.nan)
        else:
            scores.append(
                roc_auc_score(
                    y_true[:, i],
                    y_prob[:, i],
                )
            )

    return np.asarray(scores, dtype=np.float64)


def per_label_f1(y_true, y_prob, threshold=0.5):
    """Calculate F1 for each label using a fixed probability threshold."""
    y_pred = (y_prob >= threshold).astype(np.int32)

    scores = []

    for i in range(y_true.shape[1]):
        scores.append(
            f1_score(
                y_true[:, i],
                y_pred[:, i],
                zero_division=0,
            )
        )

    return np.asarray(scores, dtype=np.float64)


def tier_metrics(scores, dataset):
    """Average per-label scores within each frequency tier."""
    results = {}

    for tier in ["common", "medium", "rare"]:
        mask = dataset.tier_mask(tier)
        tier_scores = scores[mask]

        if len(tier_scores) == 0:
            results[tier] = float("nan")
        else:
            results[tier] = float(np.nanmean(tier_scores))

    return results


def evaluate(model, loader, device):
    """Run inference and collect targets and probabilities."""
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
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    return y_true, y_prob


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a PTB-XL checkpoint on the official test fold."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data-loader workers.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to calculate F1.",
    )

    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best_model.pt",
        help="Checkpoint filename inside the checkpoints directory.",
    )

    parser.add_argument(
        "--predictions-name",
        type=str,
        default="baseline_test_predictions.npz",
        help="Output filename for test predictions.",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")

    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1.")

    if Path(args.checkpoint_name).name != args.checkpoint_name:
        parser.error("--checkpoint-name must be a filename, not a path.")

    if Path(args.predictions_name).name != args.predictions_name:
        parser.error("--predictions-name must be a filename, not a path.")

    checkpoint_path = CHECKPOINT_DIR / args.checkpoint_name
    predictions_path = RESULTS_DIR / args.predictions_name

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    baseline_predictions_path = (
        RESULTS_DIR / "baseline_test_predictions.npz"
    )

    if (
        predictions_path.resolve() == baseline_predictions_path.resolve()
        and args.checkpoint_name != "best_model.pt"
    ):
        raise ValueError(
            "Only best_model.pt may write to "
            "baseline_test_predictions.npz."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("Loading official PTB-XL test fold...")
    test_ds = PTBXLDataset(split="test")

    print(
        f"Test: {len(test_ds)}  "
        f"Labels: {len(test_ds.vocab)}"
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    checkpoint_vocab = np.asarray(checkpoint["vocab"]).astype(str)
    dataset_vocab = np.asarray(test_ds.vocab).astype(str)

    if not np.array_equal(checkpoint_vocab, dataset_vocab):
        raise ValueError(
            "Checkpoint vocabulary does not match "
            "the test dataset vocabulary."
        )

    model = ResNet1D(
        n_labels=len(checkpoint["vocab"])
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])

    print("Checkpoint loaded successfully.")
    print("Running test inference...")

    y_true, y_prob = evaluate(
        model,
        test_loader,
        device,
    )

    print(f"Collected predictions: {y_prob.shape}")

    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"Target and prediction shapes differ: "
            f"{y_true.shape} vs {y_prob.shape}"
        )

    if y_prob.shape[1] != len(test_ds.vocab):
        raise ValueError(
            "Prediction label count does not match dataset vocabulary."
        )

    auroc_per_label = per_label_auroc(y_true, y_prob)
    macro_auroc = float(np.nanmean(auroc_per_label))
    auroc_tiers = tier_metrics(auroc_per_label, test_ds)

    f1_per_label_scores = per_label_f1(
        y_true,
        y_prob,
        threshold=args.threshold,
    )

    macro_f1 = float(np.nanmean(f1_per_label_scores))
    f1_tiers = tier_metrics(f1_per_label_scores, test_ds)

    print("\n" + "=" * 60)
    print("PTB-XL TEST RESULTS")
    print(f"Checkpoint: {args.checkpoint_name}")
    print(f"F1 threshold: {args.threshold}")
    print("=" * 60)

    print("\nAUROC")
    print(f"  Macro : {macro_auroc:.4f}")
    print(f"  Common: {auroc_tiers['common']:.4f}")
    print(f"  Medium: {auroc_tiers['medium']:.4f}")
    print(f"  Rare  : {auroc_tiers['rare']:.4f}")

    print("\nF1")
    print(f"  Macro : {macro_f1:.4f}")
    print(f"  Common: {f1_tiers['common']:.4f}")
    print(f"  Medium: {f1_tiers['medium']:.4f}")
    print(f"  Rare  : {f1_tiers['rare']:.4f}")

    print("=" * 60)

    np.savez_compressed(
        predictions_path,
        y_true=y_true,
        y_prob=y_prob,
        vocab=np.asarray(test_ds.vocab),
    )

    print(f"\nSaved predictions to: {predictions_path}")


if __name__ == "__main__":
    main()
