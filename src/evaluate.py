"""
Evaluate the frozen best PTB-XL model on the official test fold.

Reports:
- Macro AUROC
- Per-tier AUROC
- Macro F1
- Per-tier F1

Also saves raw probabilities and targets for downstream
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
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
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
                    y_prob[:, i]
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
                zero_division=0
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
                non_blocking=True
            )

            logits = model(signals)
            probs = torch.sigmoid(logits)

            all_probs.append(
                probs.cpu().numpy()
            )

            all_targets.append(
                targets.numpy()
            )

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    return y_true, y_prob


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5
    )

    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

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
        pin_memory=True
    )

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model = ResNet1D(
        n_labels=len(checkpoint["vocab"])
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    print("Checkpoint loaded successfully.")
    print("Running test inference...")

    y_true, y_prob = evaluate(
        model,
        test_loader,
        device
    )

    print(
        f"Collected predictions: "
        f"{y_prob.shape}"
    )

    # AUROC
    auroc_per_label = per_label_auroc(
        y_true,
        y_prob
    )

    macro_auroc = float(
        np.nanmean(auroc_per_label)
    )

    auroc_tiers = tier_metrics(
        auroc_per_label,
        test_ds
    )

    # F1
    f1_per_label = per_label_f1(
        y_true,
        y_prob,
        threshold=args.threshold
    )

    macro_f1 = float(
        np.nanmean(f1_per_label)
    )

    f1_tiers = tier_metrics(
        f1_per_label,
        test_ds
    )

    print("\n" + "=" * 60)
    print("PTB-XL TEST RESULTS")
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

    # Save raw outputs for future experiments.
    np.savez_compressed(
        RESULTS_DIR / "baseline_test_predictions.npz",
        y_true=y_true,
        y_prob=y_prob,
        vocab=np.asarray(test_ds.vocab)
    )

    print(
        f"\nSaved predictions to: "
        f"{RESULTS_DIR / 'baseline_test_predictions.npz'}"
    )


if __name__ == "__main__":
    main()