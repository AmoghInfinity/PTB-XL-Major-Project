from pathlib import Path
import json
import hashlib
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "ptbxl"
META = DATA_DIR / "ptbxl_database.csv"
SCP = DATA_DIR / "scp_statements.csv"
CHECKPOINT_DIR = ROOT / "checkpoints"
EXPECTED = 21837


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    report = {"project_root": str(ROOT), "expected_canonical_rows": EXPECTED}
    print("=== PTB-XL METADATA AUDIT ===")
    if not META.exists():
        print(f"ERROR: metadata file not found: {META}")
        print("Put this script in the project root, next to src/, data/, and checkpoints/.")
        return 2

    df = pd.read_csv(META)
    report["metadata_path"] = str(META)
    report["metadata_rows"] = int(len(df))
    report["metadata_columns"] = list(df.columns)
    print(f"Metadata file: {META}")
    print(f"Rows: {len(df)} (canonical reference used by project checker: {EXPECTED})")
    print(f"Difference: {len(df) - EXPECTED:+d}")

    required = ["ecg_id", "patient_id", "strat_fold", "filename_lr", "scp_codes"]
    missing_columns = [c for c in required if c not in df.columns]
    report["missing_required_columns"] = missing_columns
    if missing_columns:
        print("Missing columns:", missing_columns)
    else:
        print("Fold counts:")
        fold_counts = df["strat_fold"].value_counts(dropna=False).sort_index().to_dict()
        report["fold_counts"] = {str(k): int(v) for k, v in fold_counts.items()}
        print(df["strat_fold"].value_counts(dropna=False).sort_index().to_string())
        print("Duplicate ecg_id rows:", int(df["ecg_id"].duplicated().sum()))
        print("Duplicate filename_lr rows:", int(df["filename_lr"].duplicated().sum()))
        print("Missing patient_id:", int(df["patient_id"].isna().sum()))
        print("Missing strat_fold:", int(df["strat_fold"].isna().sum()))
        print("Fold values outside 1..10:", int((~df["strat_fold"].isin(range(1, 11))).sum()))
        report["duplicate_ecg_id"] = int(df["ecg_id"].duplicated().sum())
        report["duplicate_filename_lr"] = int(df["filename_lr"].duplicated().sum())
        report["missing_patient_id"] = int(df["patient_id"].isna().sum())
        report["missing_strat_fold"] = int(df["strat_fold"].isna().sum())
        report["invalid_fold_values"] = int((~df["strat_fold"].isin(range(1, 11))).sum())
        split_sizes = {
            "train_folds_1_8": int((df["strat_fold"] <= 8).sum()),
            "validation_fold_9": int((df["strat_fold"] == 9).sum()),
            "test_fold_10": int((df["strat_fold"] == 10).sum()),
        }
        report["split_sizes"] = split_sizes
        print("Split sizes:", split_sizes)
        print("Unique patients:", int(df["patient_id"].nunique(dropna=True)))
        report["unique_patients"] = int(df["patient_id"].nunique(dropna=True))

    if SCP.exists():
        scp_df = pd.read_csv(SCP)
        report["scp_statements_rows"] = int(len(scp_df))
        report["scp_statements_sha256"] = sha256(SCP)
        print(f"scp_statements.csv rows: {len(scp_df)}")
    else:
        print(f"WARNING: scp_statements.csv not found at {SCP}")

    print("\n=== CHECKPOINT METADATA AUDIT ===")
    checkpoints = ["best_model.pt", "ecg_mamba_best.pt"]
    ckpt_report = {}
    for name in checkpoints:
        path = CHECKPOINT_DIR / name
        if not path.exists():
            print(f"NOT FOUND: {path}")
            continue
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        keys = sorted(ckpt.keys()) if isinstance(ckpt, dict) else []
        info = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "top_level_keys": keys,
        }
        for key in ["model_name", "loss", "epoch", "best_val_auroc", "seed", "lr", "batch_size", "vocab"]:
            if isinstance(ckpt, dict) and key in ckpt:
                val = ckpt[key]
                if key == "vocab":
                    info["vocab_size"] = len(val)
                    info["vocab_first_10"] = list(val)[:10]
                elif isinstance(val, (str, int, float, bool)) or val is None:
                    info[key] = val
        ckpt_report[name] = info
        print(f"\n{name}")
        print("  SHA-256:", info["sha256"])
        print("  Keys:", keys)
        for k, v in info.items():
            if k not in {"path", "size_bytes", "sha256", "top_level_keys"}:
                print(f"  {k}: {v}")
    report["checkpoints"] = ckpt_report

    out = ROOT / "ptbxl_issue_audit.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved machine-readable report: {out}")
    print("\nImportant: this audit detects missing/duplicate metadata rows; it does not restore missing records.")
    print("Checkpoint metadata may not contain training seed/epoch if the training script did not save them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
