
# importing libraries
import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from explainability import (
    CHECKPOINT_DIR,
    RESULTS_DIR,
    LEAD_NAMES,
    get_device,
    load_model,
    integrated_gradients,
    calculate_lead_importance,
    save_comparison_plot,
)
from data.dataset import PTBXLDataset


def save_json(path, data):
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    temporary_path.replace(path)


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def select_pairs(dataset, labels_per_tier, examples_per_label, seed):
    rng = random.Random(seed)

    targets = np.stack([
        dataset._build_target(codes)
        for codes in dataset.scp_codes_list
    ])

    pairs = []

    for tier in ("common", "medium", "rare"):
        candidates = [
            index
            for index, label in enumerate(dataset.vocab)
            if dataset.tier_of.get(label) == tier
            and targets[:, index].sum() > 0
        ]

        candidates.sort()

        selected_labels = rng.sample(
            candidates,
            min(labels_per_tier, len(candidates)),
        )

        for label_idx in selected_labels:
            label = dataset.vocab[label_idx]

            positive_indices = np.flatnonzero(
                targets[:, label_idx] == 1
            ).tolist()

            selected_indices = rng.sample(
                positive_indices,
                min(examples_per_label, len(positive_indices)),
            )

            for record_idx in selected_indices:
                pairs.append({
                    "record_index": int(record_idx),
                    "label": str(label),
                    "label_index": int(label_idx),
                    "tier": tier,
                    "ground_truth": True,
                })

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Run paired Integrated Gradients explanations."
    )
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--labels-per-tier", type=int, default=5)
    parser.add_argument("--examples-per-label", type=int, default=10)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--alpha-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resnet-checkpoint",
        default="best_model.pt",
    )
    parser.add_argument(
        "--mamba-checkpoint",
        default="ecg_mamba_best.pt",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed pairs from a compatible previous run.",
    )
    args = parser.parse_args()

    if args.labels_per_tier < 1 or args.examples_per_label < 1:
        parser.error("Labels and examples per label must be positive.")

    if args.steps < 1 or args.alpha_batch_size < 1:
        parser.error("Steps and alpha batch size must be positive.")

    for filename in (
        args.resnet_checkpoint,
        args.mamba_checkpoint,
    ):
        if Path(filename).name != filename:
            parser.error("Checkpoint arguments must be filenames.")

    output_dir = RESULTS_DIR / "full_run"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "batch_summary.json"
    csv_path = output_dir / "lead_importance.csv"

    config = {
        "split": args.split,
        "labels_per_tier": args.labels_per_tier,
        "examples_per_label": args.examples_per_label,
        "steps": args.steps,
        "alpha_batch_size": args.alpha_batch_size,
        "seed": args.seed,
        "resnet_checkpoint": args.resnet_checkpoint,
        "mamba_checkpoint": args.mamba_checkpoint,
        "method": "Integrated Gradients",
        "baseline": "zero in normalized input space",
    }

    if args.resume and summary_path.exists():
        previous = json.loads(
            summary_path.read_text(encoding="utf-8")
        )

        if previous["config"] != config:
            raise ValueError(
                "Existing run configuration differs. "
                "Use the original settings to resume, or start a "
                "new run with a different output directory."
            )

        completed = previous["pairs"]
    else:
        completed = []

    completed_keys = {
        (item["record_index"], item["label"])
        for item in completed
    }

    device = get_device()
    dataset = PTBXLDataset(split=args.split)

    pairs = select_pairs(
        dataset,
        args.labels_per_tier,
        args.examples_per_label,
        args.seed,
    )

    print(f"Test recordings: {len(dataset)}")
    print(f"Selected recording-label pairs: {len(pairs)}")
    print(f"Already completed: {len(completed_keys)}")

    models = {
        "ResNet1D": load_model(
            "resnet1d",
            args.resnet_checkpoint,
            dataset,
            device,
        ),
        "ECG-Mamba": load_model(
            "ecg_mamba",
            args.mamba_checkpoint,
            dataset,
            device,
        ),
    }

    signal_cache = {}

    for pair_number, pair in enumerate(pairs, start=1):
        record_idx = pair["record_index"]
        label = pair["label"]
        key = (record_idx, label)

        if key in completed_keys:
            continue

        if record_idx not in signal_cache:
            signal, _ = dataset[record_idx]
            signal_cache[record_idx] = signal.unsqueeze(0)

        signal = signal_cache[record_idx].to(
            device=device,
            dtype=torch.float32,
        )

        explanations = {}
        model_summaries = {}

        for model_name, model in models.items():
            result = integrated_gradients(
                model,
                signal,
                pair["label_index"],
                args.steps,
                alpha_batch_size=args.alpha_batch_size,
            )

            importance, percentages = calculate_lead_importance(
                result["attribution"]
            )

            result["lead_importance"] = importance.tolist()
            result["lead_importance_percent"] = percentages.tolist()

            explanations[model_name] = result

            model_summaries[model_name] = {
                key_name: value
                for key_name, value in result.items()
                if key_name != "attribution"
            }

        stem = (
            f"record_{record_idx:04d}_"
            f"{safe_name(label)}"
        )

        plot_path = output_dir / f"{stem}.png"

        save_comparison_plot(
            signal,
            explanations,
            label,
            plot_path,
        )

        completed.append({
            "record_index": record_idx,
            "record_path": str(dataset.record_paths[record_idx]),
            "label": label,
            "label_index": pair["label_index"],
            "tier": pair["tier"],
            "ground_truth": pair["ground_truth"],
            "models": model_summaries,
            "plot_path": str(plot_path),
        })

        save_json(
            summary_path,
            {
                "config": config,
                "selected_pair_count": len(pairs),
                "completed_pair_count": len(completed),
                "pairs": completed,
            },
        )

        print(
            f"[{pair_number}/{len(pairs)}] "
            f"record={record_idx}, label={label}, "
            f"ResNet error="
            f"{model_summaries['ResNet1D']['absolute_completeness_error']:.5f}, "
            f"Mamba error="
            f"{model_summaries['ECG-Mamba']['absolute_completeness_error']:.5f}"
        )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "record_index",
            "label",
            "tier",
            "model",
            "lead",
            "absolute_attribution",
            "importance_percent",
        ])

        for pair in completed:
            for model_name, result in pair["models"].items():
                for index, lead in enumerate(LEAD_NAMES):
                    writer.writerow([
                        pair["record_index"],
                        pair["label"],
                        pair["tier"],
                        model_name,
                        lead,
                        result["lead_importance"][index],
                        result["lead_importance_percent"][index],
                    ])

    print("\nBatch explainability run completed.")
    print(f"Completed pairs: {len(completed)}")
    print(f"Summary: {summary_path}")
    print(f"Lead importance: {csv_path}")
    print(f"Plots: {output_dir}")


if __name__ == "__main__":
    main()
