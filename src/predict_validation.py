
# importing libraries
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import PTBXLDataset
from models.ecg_mamba import ECGMamba


def main():
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "ecg_mamba_best.pt"
    output_path = PROJECT_ROOT / "results" / "ecg_mamba_validation_predictions.npz"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    dataset = PTBXLDataset(split="val")
    checkpoint_vocab = np.asarray(checkpoint["vocab"]).astype(str)
    dataset_vocab = np.asarray(dataset.vocab).astype(str)

    if not np.array_equal(checkpoint_vocab, dataset_vocab):
        raise ValueError("Checkpoint vocabulary does not match validation vocabulary.")

    model_name = checkpoint.get("model_name", "resnet1d")
    if model_name != "ecg_mamba":
        raise ValueError(f"Expected ECG-Mamba checkpoint, got {model_name!r}.")

    model = ECGMamba(n_labels=len(dataset.vocab)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for signals, targets in loader:
            signals = signals.to(device, non_blocking=True)
            logits = model(signals)
            probabilities = torch.sigmoid(logits)

            all_targets.append(targets.numpy())
            all_probabilities.append(probabilities.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probabilities, axis=0)

    if y_true.shape != y_prob.shape:
        raise ValueError(f"Target/prediction shape mismatch: {y_true.shape}, {y_prob.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        y_true=y_true,
        y_prob=y_prob,
        vocab=np.asarray(dataset.vocab),
    )

    print("Validation examples:", len(dataset))
    print("Prediction shape:", y_prob.shape)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
