# importing libraries
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import PTBXLDataset
from models.resnet1d import ResNet1D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model_weighted_bce.pt"
OUTPUT_PATH = PROJECT_ROOT / "results" / "weighted_bce_test_results.json"


def evaluate_tier(y_true, y_prob, y_pred, mask):
    indices = np.where(mask)[0]

    auroc_values = []
    auprc_values = []
    f1_values = []

    for idx in indices:
        if len(np.unique(y_true[:, idx])) >= 2:
            auroc_values.append(
                roc_auc_score(
                    y_true[:, idx],
                    y_prob[:, idx],
                )
            )

            auprc_values.append(
                average_precision_score(
                    y_true[:, idx],
                    y_prob[:, idx],
                )
            )

        f1_values.append(
            f1_score(
                y_true[:, idx],
                y_pred[:, idx],
                zero_division=0,
            )
        )

    return {
        "auroc": float(np.mean(auroc_values)),
        "auprc": float(np.mean(auprc_values)),
        "f1": float(np.mean(f1_values)),
        "num_labels": len(indices),
    }


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    print("Loading test dataset...")

    test_ds = PTBXLDataset(split="test")

    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(
        f"Test samples: {len(test_ds)}"
    )
    print(
        f"Labels: {len(test_ds.vocab)}"
    )

    print(
        f"\nLoading checkpoint:\n"
        f"{CHECKPOINT_PATH}"
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
        test_ds.vocab
    ).astype(str)

    if not np.array_equal(
        checkpoint_vocab,
        dataset_vocab,
    ):
        raise ValueError(
            "Checkpoint vocabulary does not match "
            "the current dataset vocabulary."
        )

    model = ResNet1D(
        n_labels=len(test_ds.vocab)
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    print("Generating test predictions...")

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for signals, targets in test_loader:
            signals = signals.to(
                device,
                non_blocking=True,
            )

            logits = model(signals)

            probabilities = torch.sigmoid(
                logits
            ).cpu().numpy()

            all_probs.append(
                probabilities
            )
            all_targets.append(
                targets.numpy()
            )

    y_prob = np.concatenate(
        all_probs,
        axis=0,
    )

    y_true = np.concatenate(
        all_targets,
        axis=0,
    )

    y_pred = (
        y_prob >= 0.5
    ).astype(np.int32)

    print(
        f"Prediction shape: {y_prob.shape}"
    )

    macro_auroc = roc_auc_score(
        y_true,
        y_prob,
        average="macro",
    )

    macro_auprc = average_precision_score(
        y_true,
        y_prob,
        average="macro",
    )

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
        tier_results[tier] = evaluate_tier(
            y_true,
            y_prob,
            y_pred,
            test_ds.tier_mask(tier),
        )

    print("\nWeighted BCE test results")
    print("=========================")
    print(
        f"Macro AUROC: {macro_auroc:.4f}"
    )
    print(
        f"Macro AUPRC: {macro_auprc:.4f}"
    )
    print(
        f"Macro F1:    {macro_f1:.4f}"
    )
    print(
        f"Micro F1:    {micro_f1:.4f}"
    )

    for tier, result in tier_results.items():
        print(
            f"{tier.capitalize():8s} "
            f"AUROC={result['auroc']:.4f} "
            f"AUPRC={result['auprc']:.4f} "
            f"F1={result['f1']:.4f}"
        )

    results = {
        "checkpoint": str(CHECKPOINT_PATH),
        "test_samples": int(len(test_ds)),
        "num_labels": int(len(test_ds.vocab)),
        "threshold": 0.5,
        "macro_auroc": float(macro_auroc),
        "macro_auprc": float(macro_auprc),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "tier_results": tier_results,
        "vocab": test_ds.vocab,
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

    np.savez_compressed(
        PROJECT_ROOT
        / "results"
        / "weighted_bce_test_predictions.npz",
        y_true=y_true.astype(np.float32),
        y_prob=y_prob.astype(np.float32),
        vocab=np.asarray(
            test_ds.vocab
        ),
    )

    print(
        f"\nSaved results to: {OUTPUT_PATH}"
    )

    print(
        "Saved predictions to: "
        f"{PROJECT_ROOT / 'results' / 'weighted_bce_test_predictions.npz'}"
    )


if __name__ == "__main__":
    main()