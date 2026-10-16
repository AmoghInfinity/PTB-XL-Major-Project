# importing libraries
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import PTBXLDataset
from models.ecg_mamba import ECGMamba


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    device = torch.device("cuda")
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "ecg_mamba_best.pt"

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
    model.to(device).eval()

    signals = torch.stack([dataset[i][0] for i in range(8)]).to(device)

    timings = {"encoder": [], "block_1": [], "block_2": [],
               "block_3": [], "block_4": [], "head": []}

    def measure(name, operation):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = operation()
        torch.cuda.synchronize()
        timings[name].append((time.perf_counter() - start) * 1000)
        return result

    with torch.inference_mode():
        for _ in range(10):
            model(signals)

        for _ in range(30):
            x = measure("encoder", lambda: model.encoder(signals))
            x = x.transpose(1, 2)

            for index, block in enumerate(model.blocks, start=1):
                x = measure(f"block_{index}", lambda block=block, x=x: block(x))

            x = measure("head", lambda: model.classifier(model.norm(x).mean(dim=1)))

    print("\nECG-Mamba component timing, batch size 8")
    print("Each component is measured independently; totals may differ")
    print("from full-model latency due to measurement overhead.\n")

    for name, values in timings.items():
        values.sort()
        median = values[len(values) // 2]
        mean = sum(values) / len(values)
        print(f"{name:10s} mean={mean:8.3f} ms  median={median:8.3f} ms")

    print("\nComponent profiling completed.")


if __name__ == "__main__":
    main()
