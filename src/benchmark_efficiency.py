
# importing libraries
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from data.dataset import PTBXLDataset
from models.resnet1d import ResNet1D
from models.ecg_mamba import ECGMamba


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "results" / "efficiency"

MODELS = {
    "ResNet1D": (ResNet1D, "best_model.pt"),
    "ECG-Mamba": (ECGMamba, "ecg_mamba_best.pt"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PTB-XL model inference efficiency."
    )
    parser.add_argument(
        "--split", choices=("train", "val", "test"), default="test"
    )
    parser.add_argument(
        "--batch-sizes", default="1,8,16,32,64,128"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto"
    )
    return parser.parse_args()


def get_device(requested):
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable.")
        return torch.device("cuda")

    if requested == "cpu":
        return torch.device("cpu")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def load_model(name, dataset, device):
    model_class, filename = MODELS[name]
    checkpoint_path = CHECKPOINT_DIR / filename

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint_vocab = checkpoint.get("vocab")
    if (
        checkpoint_vocab is not None
        and list(checkpoint_vocab) != list(dataset.vocab)
    ):
        raise ValueError(
            f"Label vocabulary mismatch in {checkpoint_path}"
        )

    model = model_class(n_labels=len(dataset.vocab))

    if "model_state" not in checkpoint:
        raise KeyError(
            f"'model_state' missing from {checkpoint_path}"
        )

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    checkpoint_size_mb = (
        checkpoint_path.stat().st_size / (1024 ** 2)
    )

    del checkpoint

    return (
        model,
        parameter_count,
        checkpoint_size_mb,
        checkpoint_path,
    )


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_batch(
    model, inputs, batch_size, warmup, repeats, device
):
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)

        synchronize(device)

        baseline_memory = (
            torch.cuda.memory_allocated(device)
            if device.type == "cuda"
            else None
        )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        iteration_times_ms = []

        for _ in range(repeats):
            synchronize(device)
            start = time.perf_counter()

            model(inputs)

        synchronize(device)
        iteration_times_ms.append(
            (time.perf_counter() - start) * 1000
        )

    latency_batch_ms = sum(iteration_times_ms) / repeats
    latency_ecg_ms = latency_batch_ms / batch_size
    throughput = batch_size / (elapsed / repeats)

    peak_memory_mb = None
    incremental_memory_mb = None

    if device.type == "cuda":
        peak_bytes = torch.cuda.max_memory_allocated(device)

        peak_memory_mb = peak_bytes / (1024 ** 2)
        incremental_memory_mb = (
            max(0, peak_bytes - baseline_memory) / (1024 ** 2)
        )

    return {
        "status": "ok",
        "repetitions": repeats,
        "total_elapsed_seconds": elapsed,
        "latency_per_batch_ms": latency_batch_ms,
        "latency_per_ecg_ms": latency_ecg_ms,
        "throughput_ecgs_per_second": throughput,
        "peak_gpu_memory_mb": peak_memory_mb,
        "incremental_peak_gpu_memory_mb": incremental_memory_mb,
        "median_latency_per_batch_ms": float(
            torch.tensor(iteration_times_ms).median().item()
        ),
        "p90_latency_per_batch_ms": float(
            torch.quantile(
                torch.tensor(iteration_times_ms), 0.90
            ).item()
        ),
    }


def main():
    args = parse_args()

    if args.warmup < 0 or args.repeats < 1:
        raise ValueError(
            "--warmup must be >= 0 and --repeats must be >= 1."
        )

    try:
        batch_sizes = sorted(
            {int(value.strip())
             for value in args.batch_sizes.split(",")}
        )
    except ValueError as error:
        raise ValueError(
            "--batch-sizes must contain comma-separated integers."
        ) from error

    if not batch_sizes or any(size < 1 for size in batch_sizes):
        raise ValueError("Batch sizes must be positive integers.")

    device = get_device(args.device)

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        print(f"Using GPU: {device_name}")
    else:
        device_name = "CPU"
        print("Using CPU")

    print(f"Loading PTB-XL {args.split} split...")
    dataset = PTBXLDataset(split=args.split)

    sample_count = min(max(batch_sizes), len(dataset))
    signals = []

    for index in range(sample_count):
        item = dataset[index]
        signal = item[0] if isinstance(item, (tuple, list)) else item

        if not torch.is_tensor(signal):
            signal = torch.as_tensor(signal)

        if signal.ndim != 2:
            raise ValueError(
                f"Expected ECG shape (12, 1000), got "
                f"{tuple(signal.shape)} at index {index}."
            )

        signals.append(
            signal.detach().to(device="cpu", dtype=torch.float32)
        )

    inputs_cpu = torch.stack(signals, dim=0)

    print(f"ECGs loaded: {len(inputs_cpu)}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Warm-up passes: {args.warmup}")
    print(f"Timed passes: {args.repeats}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for model_name in MODELS:
        print(f"\nLoading {model_name}...")

        model, parameters, checkpoint_mb, checkpoint_path = (
            load_model(model_name, dataset, device)
        )

        print(f"Checkpoint: {checkpoint_path.name}")
        print(f"Parameters: {parameters:,}")

        oom_encountered = False

        for batch_size in batch_sizes:
            base = {
                "model": model_name,
                "split": args.split,
                "batch_size": batch_size,
                "device": str(device),
                "parameter_count": parameters,
                "checkpoint_size_mb": checkpoint_mb,
            }

            if batch_size > len(inputs_cpu):
                results.append({
                    **base,
                    "status": "skipped: insufficient recordings",
                })
                print(f"  Batch {batch_size}: skipped")
                continue

            if oom_encountered:
                results.append({
                    **base,
                    "status": "skipped: previous CUDA OOM",
                })
                continue

            inputs = inputs_cpu[:batch_size].to(device)

            try:
                if device.type == "cuda":
                    synchronize(device)
                    torch.cuda.empty_cache()

                measurement = benchmark_batch(
                    model,
                    inputs,
                    batch_size,
                    args.warmup,
                    args.repeats,
                    device,
                )

                results.append({**base, **measurement})

                memory = measurement["peak_gpu_memory_mb"]
                memory_text = (
                    f", peak GPU memory={memory:.1f} MB"
                    if memory is not None
                    else ""
                )

                print(
                    f"  Batch {batch_size}: "
                    f"{measurement['latency_per_batch_ms']:.3f} "
                    f"ms/batch, "
                    f"{measurement['latency_per_ecg_ms']:.3f} "
                    f"ms/ECG, "
                    f"{measurement['throughput_ecgs_per_second']:.2f} "
                    f"ECG/s{memory_text}"
                )

            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise

                oom_encountered = True

                if device.type == "cuda":
                    torch.cuda.empty_cache()

                results.append({
                    **base,
                    "status": "skipped: CUDA out of memory",
                })

                print(
                    f"  Batch {batch_size}: CUDA OOM; "
                    "larger batches will be skipped."
                )

            finally:
                del inputs

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    columns = [
        "model",
        "split",
        "batch_size",
        "device",
        "status",
        "repetitions",
        "total_elapsed_seconds",
        "latency_per_batch_ms",
        "latency_per_ecg_ms",
        "throughput_ecgs_per_second",
        "parameter_count",
        "checkpoint_size_mb",
        "peak_gpu_memory_mb",
        "incremental_peak_gpu_memory_mb",
    ]

    csv_path = OUTPUT_DIR / "benchmark_efficiency.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(results)

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "split": args.split,
        "device": str(device),
        "device_name": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "warmup_passes": args.warmup,
        "timed_passes": args.repeats,
        "batch_sizes": batch_sizes,
        "recordings_loaded": len(inputs_cpu),
        "timing_scope": (
            "Model forward pass on device-resident ECG tensors. "
            "Dataset loading, checkpoint loading, warm-up, and "
            "host-to-device transfer are excluded."
        ),
        "results": results,
    }

    json_path = OUTPUT_DIR / "benchmark_efficiency.json"
    json_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nBenchmark completed.")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
