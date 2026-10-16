# importing libraries
import sys
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import PTBXLDataset
from models.ecg_mamba import ECGMamba


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This profile requires a GPU.")

    device = torch.device("cuda")
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "ecg_mamba_best.pt"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Loading ECG-Mamba checkpoint...")

    dataset = PTBXLDataset(split="test")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "vocab" in checkpoint and list(checkpoint["vocab"]) != list(dataset.vocab):
        raise ValueError("Checkpoint vocabulary does not match the dataset.")

    model = ECGMamba(n_labels=len(dataset.vocab))
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )
    signals, _ = next(iter(loader))
    signals = signals.to(device)

    print(f"Input shape: {tuple(signals.shape)}")
    print("Running warm-up...")

    with torch.inference_mode():
        for _ in range(5):
            model(signals)

    torch.cuda.synchronize()

    activities = [
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ]

    print("Profiling three forward passes...")

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with torch.inference_mode():
            for _ in range(3):
                model(signals)
                torch.cuda.synchronize()

    print("\nTOP OPERATIONS BY CUDA TIME")
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=20,
        )
    )

    print("\nTOP OPERATIONS BY CPU TIME")
    print(
        prof.key_averages().table(
            sort_by="self_cpu_time_total",
            row_limit=20,
        )
    )

    print("\nProfiling completed.")


if __name__ == "__main__":
    main()
