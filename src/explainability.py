
# importing libraries
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from data.dataset import PTBXLDataset
from models.resnet1d import ResNet1D
from models.ecg_mamba import ECGMamba


CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "explainability"
    / "full_run_corrected"
)

LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device

    print("WARNING: CUDA unavailable. Using CPU.")
    return torch.device("cpu")


def load_model(model_name, checkpoint_name, dataset, device):
    checkpoint_path = CHECKPOINT_DIR / checkpoint_name

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    checkpoint_vocab = np.asarray(checkpoint["vocab"]).astype(str)
    dataset_vocab = np.asarray(dataset.vocab).astype(str)

    if not np.array_equal(checkpoint_vocab, dataset_vocab):
        raise ValueError(
            f"{model_name}: checkpoint vocabulary does not match "
            "the dataset vocabulary."
        )

    saved_model_name = checkpoint.get("model_name", "resnet1d")
    if saved_model_name != model_name:
        raise ValueError(
            f"Expected checkpoint for {model_name}, "
            f"but found {saved_model_name}."
        )

    if model_name == "resnet1d":
        model = ResNet1D(n_labels=len(checkpoint_vocab))
    elif model_name == "ecg_mamba":
        model = ECGMamba(n_labels=len(checkpoint_vocab))
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model = model.to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Loaded {model_name}: {checkpoint_path.name}")
    return model




def integrated_gradients(
    model,
    signal,
    label_idx,
    steps,
    alpha_batch_size=8,
):
    """Calculate Integrated Gradients using batched interpolation points."""

    if steps < 1:
        raise ValueError("steps must be at least 1.")

    if alpha_batch_size < 1:
        raise ValueError("alpha_batch_size must be at least 1.")

    baseline = torch.zeros_like(signal)
    difference = signal - baseline

    alphas = torch.linspace(
        0.0,
        1.0,
        steps + 1,
        device=signal.device,
        dtype=signal.dtype,
    )

    gradient_sum = torch.zeros_like(signal)

    for start in range(0, len(alphas), alpha_batch_size):
        alpha_chunk = alphas[start:start + alpha_batch_size]

        interpolated = (
            baseline
            + alpha_chunk[:, None, None] * difference
        ).detach().requires_grad_(True)

        logits = model(interpolated)
        target_logits = logits[:, label_idx]

        gradients = torch.autograd.grad(
            outputs=target_logits.sum(),
            inputs=interpolated,
            create_graph=False,
            retain_graph=False,
        )[0]

        global_indices = torch.arange(
            start,
            start + len(alpha_chunk),
        device=signal.device,
        )

        weights = torch.ones_like(alpha_chunk)
        weights[global_indices == 0] = 0.5
        weights[global_indices == steps] = 0.5

        gradient_sum += (
            gradients
            * weights[:, None, None]
        ).sum(dim=0, keepdim=True)

    average_gradient = gradient_sum / steps
    attribution = difference * average_gradient

    with torch.no_grad():
        baseline_logit = model(baseline)[0, label_idx].item()
        input_logit = model(signal)[0, label_idx].item()
        probability = torch.sigmoid(
            torch.tensor(input_logit)
        ).item()

    attribution_sum = attribution.sum().item()
    logit_difference = input_logit - baseline_logit
    completeness_residual = attribution_sum - logit_difference

    return {
        "attribution": attribution[0].detach().cpu().numpy(),
        "input_logit": float(input_logit),
        "baseline_logit": float(baseline_logit),
        "probability": float(probability),
        "attribution_sum": float(attribution_sum),
        "logit_difference": float(logit_difference),
        "completeness_residual": float(completeness_residual),
        "absolute_completeness_error": float(
            abs(completeness_residual)
        ),
    }




def calculate_lead_importance(attribution):
    """Calculate absolute attribution importance per ECG lead."""

    importance = np.abs(attribution).sum(axis=1)
    total = importance.sum()

    if total > 0:
        percentages = 100.0 * importance / total
    else:
        percentages = np.zeros_like(importance)

    return importance, percentages


def save_comparison_plot(signal, explanations, label, output_path):
    """Save paired ECG waveform plots colored by signed attribution."""

    signal = signal[0].detach().cpu().numpy()
    times = np.arange(signal.shape[1]) / 100.0

    all_attributions = np.concatenate(
        [
            item["attribution"].ravel()
            for item in explanations.values()
        ]
    )
    vmax = float(np.max(np.abs(all_attributions)))

    if vmax == 0:
        vmax = 1e-12

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")

    figure, axes = plt.subplots(
        nrows=12,
        ncols=len(explanations),
        figsize=(14, 22),
        sharex=True,
    )

    if len(explanations) == 1:
        axes = axes.reshape(12, 1)

    for column, (model_name, result) in enumerate(
        explanations.items()
    ):
        attribution = result["attribution"]

        for lead_idx, lead_name in enumerate(LEAD_NAMES):
            axis = axes[lead_idx, column]
            waveform = signal[lead_idx]
            lead_attribution = attribution[lead_idx]

            points = np.column_stack((times, waveform))
            segments = np.stack(
                [points[:-1], points[1:]],
                axis=1,
            )
            segment_values = (
                lead_attribution[:-1]
                + lead_attribution[1:]
            ) / 2.0

            line = LineCollection(
                segments,
                cmap=cmap,
                norm=norm,
                linewidth=0.8,
            )
            line.set_array(segment_values)
            axis.add_collection(line)

            axis.set_xlim(times[0], times[-1])
            amplitude_limit = max(
                float(np.max(np.abs(waveform))),
                1e-6,
            )
            axis.set_ylim(
                -1.1 * amplitude_limit,
                1.1 * amplitude_limit,
            )
            axis.set_ylabel(lead_name, rotation=0, labelpad=18)
            axis.grid(alpha=0.2)

            if lead_idx == 0:
                axis.set_title(
                    f"{model_name}\n"
                    f"Probability: {result['probability']:.4f}"
                )

            if lead_idx == 11:
                axis.set_xlabel("Time (seconds)")

    figure.suptitle(
        f"Integrated Gradients: {label}\n"
        "Red and blue indicate positive and negative attribution",
        y=0.995,
    )

    scalar_map = ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    figure.colorbar(
        scalar_map,
        ax=axes.ravel().tolist(),
        label="Signed input attribution",
        shrink=0.6,
        pad=0.02,
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare Integrated Gradients explanations for "
            "ResNet1D and ECG-Mamba on one PTB-XL recording."
        )
    )

    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--label", type=str, default="NORM")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument(
        "--resnet-checkpoint",
        default="best_model.pt",
    )
    parser.add_argument(
        "--mamba-checkpoint",
        default="ecg_mamba_best.pt",
    )

    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be at least 1.")

    for checkpoint_name in (
        args.resnet_checkpoint,
        args.mamba_checkpoint,
    ):
        if Path(checkpoint_name).name != checkpoint_name:
            parser.error("Checkpoint arguments must be filenames.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print(f"Loading PTB-XL {args.split} split...")
    dataset = PTBXLDataset(split=args.split)

    if not 0 <= args.index < len(dataset):
        raise IndexError(
            f"Index {args.index} is outside the dataset range "
            f"0 to {len(dataset) - 1}."
        )

    label_to_idx = {
        label: index
        for index, label in enumerate(dataset.vocab)
    }

    if args.label not in label_to_idx:
        raise ValueError(
            f"Unknown label: {args.label}. "
            f"Available labels: {', '.join(dataset.vocab)}"
        )

    label_idx = label_to_idx[args.label]

    signal, target = dataset[args.index]
    signal = signal.unsqueeze(0).to(device=device, dtype=torch.float32)

    if signal.shape != (1, 12, 1000):
        raise ValueError(
            f"Unexpected ECG shape: {tuple(signal.shape)}"
        )

    print(f"Recording index: {args.index}")
    print(f"Label: {args.label}")
    print(f"Ground-truth label: {bool(target[label_idx].item())}")
    print(f"Integrated Gradients steps: {args.steps}")

    model_configs = {
        "ResNet1D": (
            "resnet1d",
            args.resnet_checkpoint,
        ),
        "ECG-Mamba": (
            "ecg_mamba",
            args.mamba_checkpoint,
        ),
    }

    explanations = {}

    for display_name, (model_name, checkpoint_name) in (
        model_configs.items()
    ):
        model = load_model(
            model_name,
            checkpoint_name,
            dataset,
            device,
        )

        print(f"Calculating attribution: {display_name}")

        result = integrated_gradients(
            model,
            signal,
            label_idx,
            args.steps,
        )

        importance, percentages = calculate_lead_importance(
            result["attribution"]
        )

        result["lead_importance"] = importance.tolist()
        result["lead_importance_percent"] = percentages.tolist()
        result["checkpoint"] = checkpoint_name

        print(
            f"  Probability: {result['probability']:.6f}"
        )
        print(
            "  Absolute completeness error: "
            f"{result['absolute_completeness_error']:.6f}"
        )

        explanations[display_name] = result

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    stem = (
        f"{args.split}_index_{args.index}_"
        f"label_{args.label}"
    )

    plot_path = RESULTS_DIR / f"{stem}_comparison.png"
    save_comparison_plot(
        signal,
        explanations,
        args.label,
        plot_path,
    )

    csv_path = RESULTS_DIR / f"{stem}_lead_importance.csv"
    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "lead",
                "resnet_absolute_attribution",
                "resnet_importance_percent",
                "ecg_mamba_absolute_attribution",
                "ecg_mamba_importance_percent",
            ]
        )

        resnet = explanations["ResNet1D"]
        mamba = explanations["ECG-Mamba"]

        for index, lead in enumerate(LEAD_NAMES):
            writer.writerow(
                [
                    lead,
                    resnet["lead_importance"][index],
                    resnet["lead_importance_percent"][index],
                    mamba["lead_importance"][index],
                    mamba["lead_importance_percent"][index],
                ]
            )

    summary = {
        "split": args.split,
        "recording_index": args.index,
        "record_path": str(dataset.record_paths[args.index]),
        "label": args.label,
        "label_index": label_idx,
        "ground_truth": bool(target[label_idx].item()),
        "baseline": "zero in normalized model-input space",
        "method": "Integrated Gradients",
        "integration_steps": args.steps,
        "input_shape": list(signal.shape),
        "models": explanations,
        "plot_path": str(plot_path),
        "lead_importance_csv": str(csv_path),
    }

    json_path = RESULTS_DIR / f"{stem}_summary.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item() if isinstance(value, np.generic) else str(value))

    print("\nExplainability smoke test completed.")
    print(f"Plot: {plot_path}")
    print(f"Lead importance: {csv_path}")
    print(f"Summary: {json_path}")


if __name__ == "__main__":
    main()
