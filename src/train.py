"""
Training loop for the PTB-XL multi-label ECG classifier.
Includes AMP mixed precision (bf16 on Blackwell) and per-frequency-tier
AUROC reporting, since that breakdown is the project's core evaluation axis.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import PTBXLDataset
from models.resnet1d import ResNet1D

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"Using GPU: {name} (compute capability {cap})")
        if cap >= (12, 0):
            print("Blackwell GPU detected (sm_120) — confirm your PyTorch build")
            print("was installed from the cu128+ index, or this will silently")
            print("run on generic/slow kernels. See README.md.")
        return device
    print("WARNING: CUDA not available — training on CPU will be very slow.")
    print("If you have an RTX 5050, see README.md for the cu128 install fix.")
    return torch.device("cpu")


def per_tier_auroc(y_true: np.ndarray, y_prob: np.ndarray, tier_mask: np.ndarray) -> float:
    """Macro-AUROC restricted to the labels in one frequency tier."""
    idx = np.where(tier_mask)[0]
    if len(idx) == 0:
        return float("nan")
    aurocs = []
    for i in idx:
        if len(np.unique(y_true[:, i])) < 2:
            continue  # AUROC undefined if a label has only one class present
        aurocs.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
    return float(np.mean(aurocs)) if aurocs else float("nan")


def evaluate(model, loader, device, dataset: PTBXLDataset):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for signals, targets in loader:
            signals = signals.to(device, non_blocking=True)
            logits = model(signals)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets)

    results = {"macro_auroc": per_tier_auroc(y_true, y_prob, np.ones(y_prob.shape[1], dtype=bool))}
    for tier in ["common", "medium", "rare"]:
        results[f"{tier}_auroc"] = per_tier_auroc(y_true, y_prob, dataset.tier_mask(tier))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (recommended on Blackwell)")
    args = parser.parse_args()

    device = get_device()
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    print("Loading datasets (this parses labels and builds the vocabulary)...")
    train_ds = PTBXLDataset(split="train")
    val_ds = PTBXLDataset(split="val")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Labels: {len(train_ds.vocab)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = ResNet1D(n_labels=len(train_ds.vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    # bf16 autocast is the right choice on Blackwell (vs. fp16) — better
    # numerical stability, and Blackwell's tensor cores are built for it.
    amp_dtype = torch.bfloat16
    scaler_enabled = args.amp and device.type == "cuda"

    best_val_auroc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        for signals, targets in train_loader:
            signals = signals.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if scaler_enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(signals)
                    loss = criterion(logits, targets)
            else:
                logits = model(signals)
                loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * signals.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_ds)
        val_metrics = evaluate(model, val_loader, device, val_ds)
        elapsed = time.time() - epoch_start

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={train_loss:.4f}  "
            f"val_macro_auroc={val_metrics['macro_auroc']:.4f}  "
            f"common={val_metrics['common_auroc']:.4f}  "
            f"medium={val_metrics['medium_auroc']:.4f}  "
            f"rare={val_metrics['rare_auroc']:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if val_metrics["macro_auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["macro_auroc"]
            torch.save(
                {"model_state": model.state_dict(), "vocab": train_ds.vocab, "tier_of": train_ds.tier_of},
                CHECKPOINT_DIR / "best_model.pt",
            )
            print(f"  -> saved new best checkpoint (val_macro_auroc={best_val_auroc:.4f})")

    print(f"\nTraining complete. Best val macro-AUROC: {best_val_auroc:.4f}")
    print(f"Checkpoint saved to: {CHECKPOINT_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    main()
