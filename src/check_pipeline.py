"""
Read-only PTB-XL data and preprocessing audit.

This script does NOT modify:
- the dataset
- model checkpoints
- training code
- calibration artifacts

It checks:
1. PTB-XL metadata availability and recording counts
2. Official fold assignments
3. Patient overlap between train/validation/test
4. 100 Hz recording paths and WFDB sampling rates
5. Signal shape and NaN/Inf values
6. Band-pass filtering behavior
7. Per-lead normalization
8. SCP diagnostic-label construction
9. Label counts and positive-label coverage
10. Vocabulary/tier leakage caused by constructing labels from all folds
11. Whether train-only and all-data vocabularies differ
12. Basic preprocessing consistency across splits
"""

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.labels import (
    PTBXL_DIR,
    load_metadata,
    load_scp_statements,
    build_label_vocab,
    assign_frequency_tiers,
)

LOWCUT_HZ = 0.5
HIGHCUT_HZ = 40.0
SAMPLING_RATE = 100


def header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def ok(message):
    print(f"[PASS] {message}")


def warn(message):
    print(f"[WARN] {message}")


def fail(message):
    print(f"[FAIL] {message}")


def bandpass_filter(signal, fs=SAMPLING_RATE):
    nyq = 0.5 * fs
    b, a = butter(
        3,
        [LOWCUT_HZ / nyq, HIGHCUT_HZ / nyq],
        btype="band",
    )
    return filtfilt(b, a, signal, axis=0)


def parse_codes(value):
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        return ast.literal_eval(value)

    raise TypeError(
        f"Unsupported scp_codes type: {type(value)}"
    )


def resolve_record_path(rel_path):
    path = PTBXL_DIR / rel_path

    if path.suffix in {".dat", ".hea"}:
        path = path.with_suffix("")

    return path


def check_metadata(df):
    header("1. METADATA AND DATASET SIZE")

    required = {
        "strat_fold",
        "patient_id",
        "filename_lr",
        "scp_codes",
    }

    missing = required - set(df.columns)

    if missing:
        fail(
            f"Missing required metadata columns: "
            f"{sorted(missing)}"
        )
    else:
        ok("All required metadata columns are present.")

    print(f"Dataset metadata rows: {len(df)}")
    print(
        f"Expected PTB-XL recordings from requirements: 21837"
    )

    if len(df) == 21837:
        ok("Metadata contains 21,837 recordings.")
    else:
        warn(
            f"Metadata contains {len(df)} rows, not 21,837. "
            "Investigate before final experiments."
        )

    print("\nFold counts:")
    print(
        df["strat_fold"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    split_sizes = {
        "train": int((df["strat_fold"] <= 8).sum()),
        "val": int((df["strat_fold"] == 9).sum()),
        "test": int((df["strat_fold"] == 10).sum()),
    }

    print("\nProject split:")
    for name, count in split_sizes.items():
        print(f"  {name:5s}: {count}")

    if (
        split_sizes["train"]
        + split_sizes["val"]
        + split_sizes["test"]
        == len(df)
    ):
        ok("Every metadata row belongs to the project train/val/test split.")
    else:
        fail("Some metadata rows are outside folds 1-10.")

    return split_sizes


def check_patient_overlap(df):
    header("2. PATIENT-LEVEL SPLIT INTEGRITY")

    train_patients = set(
        df.loc[df["strat_fold"] <= 8, "patient_id"]
    )
    val_patients = set(
        df.loc[df["strat_fold"] == 9, "patient_id"]
    )
    test_patients = set(
        df.loc[df["strat_fold"] == 10, "patient_id"]
    )

    train_val = train_patients & val_patients
    train_test = train_patients & test_patients
    val_test = val_patients & test_patients

    print(f"Train patients: {len(train_patients)}")
    print(f"Val patients:   {len(val_patients)}")
    print(f"Test patients:  {len(test_patients)}")

    print(f"\nTrain ∩ Val:  {len(train_val)}")
    print(f"Train ∩ Test: {len(train_test)}")
    print(f"Val ∩ Test:   {len(val_test)}")

    if not train_val and not train_test and not val_test:
        ok("No patient overlap exists between train, validation, and test.")
    else:
        fail(
            "Patient overlap detected. This would invalidate "
            "the official patient-stratified evaluation."
        )


def check_recordings(df, sample_per_split=5):
    header("3. 100 Hz RECORDING PATHS AND SIGNAL FORMAT")

    missing = []
    samples = []

    for _, row in df.iterrows():
        path = resolve_record_path(row["filename_lr"])

        if not (
            Path(str(path) + ".dat").exists()
            and Path(str(path) + ".hea").exists()
        ):
            missing.append(str(path))

    print(f"Recording metadata rows checked: {len(df)}")
    print(f"Missing WFDB records: {len(missing)}")

    if missing:
        fail(
            "Some filename_lr entries do not have both .dat and .hea files."
        )
        print("First missing records:")
        for path in missing[:10]:
            print(f"  {path}")
    else:
        ok("Every filename_lr entry has a .dat and .hea file.")

    rng = np.random.default_rng(42)

    for split_name, mask in [
        ("train", df["strat_fold"] <= 8),
        ("val", df["strat_fold"] == 9),
        ("test", df["strat_fold"] == 10),
    ]:
        candidates = df.loc[mask]

        if len(candidates) == 0:
            continue

        n = min(sample_per_split, len(candidates))
        chosen = candidates.sample(
            n=n,
            random_state=42,
        )

        for _, row in chosen.iterrows():
            path = resolve_record_path(row["filename_lr"])

            try:
                signal, fields = wfdb.rdsamp(str(path))
                samples.append(
                    {
                        "split": split_name,
                        "ecg_id": row.name,
                        "fs": fields["fs"],
                        "shape": signal.shape,
                        "sig_len": fields["sig_len"],
                        "n_sig": fields["n_sig"],
                        "nan": bool(np.isnan(signal).any()),
                        "inf": bool(np.isinf(signal).any()),
                    }
                )
            except Exception as exc:
                fail(
                    f"Could not read ECG {row.name}: {exc}"
                )

    print("\nSampled WFDB records:")
    for item in samples:
        print(
            f"  {item['split']:5s} "
            f"ecg_id={item['ecg_id']} "
            f"fs={item['fs']} "
            f"shape={item['shape']} "
            f"NaN={item['nan']} "
            f"Inf={item['inf']}"
        )

    if samples:
        all_100hz = all(
            item["fs"] == 100
            for item in samples
        )
        all_12lead = all(
            item["shape"][1] == 12
            for item in samples
        )
        all_1000 = all(
            item["shape"][0] == 1000
            for item in samples
        )

        if all_100hz:
            ok("Sampled recordings are all 100 Hz.")
        else:
            fail("At least one sampled recording is not 100 Hz.")

        if all_12lead:
            ok("Sampled recordings all contain 12 leads.")
        else:
            fail("At least one sampled recording is not 12-lead.")

        if all_1000:
            ok("Sampled 100 Hz recordings contain 1000 samples.")
        else:
            fail(
                "At least one sampled 100 Hz recording does not contain "
                "1000 samples."
            )


def check_preprocessing(df, sample_count=9):
    header("4. PREPROCESSING AUDIT")

    sample = df.sample(
        n=min(sample_count, len(df)),
        random_state=42,
    )

    filter_errors = []
    normalization_errors = []
    shape_errors = []
    raw_stats = []
    processed_stats = []

    for _, row in sample.iterrows():
        path = resolve_record_path(row["filename_lr"])

        raw, fields = wfdb.rdsamp(str(path))

        if fields["fs"] != SAMPLING_RATE:
            continue

        if raw.shape != (1000, 12):
            shape_errors.append(
                (row.name, raw.shape)
            )
            continue

        if not np.isfinite(raw).all():
            fail(
                f"Raw signal for ECG {row.name} contains NaN/Inf."
            )
            continue

        processed = bandpass_filter(
            raw,
            fs=SAMPLING_RATE,
        )

        processed = (
            processed
            - processed.mean(axis=0, keepdims=True)
        ) / (
            processed.std(axis=0, keepdims=True)
            + 1e-8
        )

        raw_stats.append(
            (
                float(np.mean(raw)),
                float(np.std(raw)),
            )
        )

        processed_stats.append(
            (
                np.mean(processed, axis=0),
                np.std(processed, axis=0),
            )
        )

        if not np.isfinite(processed).all():
            filter_errors.append(row.name)
            continue

        if processed.shape != (1000, 12):
            shape_errors.append(
                (row.name, processed.shape)
            )

        means = processed.mean(axis=0)
        stds = processed.std(axis=0)

        if not np.all(np.abs(means) < 1e-5):
            normalization_errors.append(
                (row.name, "mean", float(np.max(np.abs(means))))
            )

        if not np.all(np.abs(stds - 1.0) < 1e-5):
            normalization_errors.append(
                (row.name, "std", float(np.max(np.abs(stds - 1.0))))
            )

    print(
        f"Filter range configured in dataset.py: "
        f"{LOWCUT_HZ}–{HIGHCUT_HZ} Hz"
    )
    print(
        f"Sampling rate configured in dataset.py: "
        f"{SAMPLING_RATE} Hz"
    )
    print("Filter: 3rd-order Butterworth + filtfilt")
    print("Normalization: per-lead mean/std")

    if filter_errors:
        fail(
            f"Non-finite values appeared after preprocessing: "
            f"{filter_errors}"
        )
    else:
        ok("Filtered signals remain finite.")

    if shape_errors:
        fail(
            f"Unexpected signal shapes: {shape_errors}"
        )
    else:
        ok("Processed sample shapes remain 1000 × 12.")

    if normalization_errors:
        fail(
            "Per-lead normalization is not producing approximately "
            "zero mean and unit standard deviation."
        )
        for item in normalization_errors[:10]:
            print(f"  {item}")
    else:
        ok(
            "Processed leads have approximately zero mean and unit "
            "standard deviation."
        )


def check_labels(df, scp_df):
    header("5. SCP-ECG LABEL CONSTRUCTION")

    if "diagnostic" not in scp_df.columns:
        fail(
            "scp_statements.csv does not contain the diagnostic column."
        )
        return None

    diagnostic_codes = set(
        scp_df.loc[
            scp_df["diagnostic"] == 1
        ].index
    )

    print(
        f"Diagnostic SCP codes in scp_statements.csv: "
        f"{len(diagnostic_codes)}"
    )

    counts = {}

    for codes in df["scp_codes"]:
        codes = parse_codes(codes)

        for code in codes:
            if code in diagnostic_codes:
                counts[code] = counts.get(code, 0) + 1

    print(
        f"Diagnostic codes observed in metadata: "
        f"{len(counts)}"
    )

    min_count = 10

    vocab = sorted(
        code
        for code, count in counts.items()
        if count >= min_count
    )

    print(
        f"Current vocabulary (min_count={min_count}): "
        f"{len(vocab)}"
    )

    print(
        f"Diagnostic codes below min_count: "
        f"{sum(count < min_count for count in counts.values())}"
    )

    unknown_codes = set()

    for codes in df["scp_codes"]:
        codes = parse_codes(codes)

        for code in codes:
            if code not in diagnostic_codes:
                unknown_codes.add(code)

    print(
        f"Non-diagnostic/form/rhythm codes encountered: "
        f"{len(unknown_codes)}"
    )

    if vocab:
        print("\nCurrent vocabulary and full-dataset counts:")
        for code in vocab:
            print(
                f"  {code:8s} "
                f"n={counts[code]:5d}"
            )

    return {
        "diagnostic_codes": diagnostic_codes,
        "counts": counts,
        "vocab": vocab,
    }


def check_split_label_distribution(df, vocab):
    header("6. LABEL DISTRIBUTION BY SPLIT")

    rows = []

    for code in vocab:
        row = {"label": code}

        for split_name, mask in [
            ("train", df["strat_fold"] <= 8),
            ("val", df["strat_fold"] == 9),
            ("test", df["strat_fold"] == 10),
        ]:
            count = 0

            for codes in df.loc[mask, "scp_codes"]:
                codes = parse_codes(codes)

                if code in codes:
                    count += 1

            row[split_name] = count

        rows.append(row)

    dist = pd.DataFrame(rows)

    print(dist.to_string(index=False))

    zero_train = dist.loc[
        dist["train"] == 0,
        "label",
    ].tolist()

    zero_val = dist.loc[
        dist["val"] == 0,
        "label",
    ].tolist()

    zero_test = dist.loc[
        dist["test"] == 0,
        "label",
    ].tolist()

    if zero_train:
        fail(
            "Labels exist in the current vocabulary but have zero "
            f"training positives: {zero_train}"
        )
    else:
        ok("Every vocabulary label has at least one training positive.")

    if zero_val:
        warn(
            "Labels have zero validation positives: "
            f"{zero_val}"
        )

    if zero_test:
        warn(
            "Labels have zero test positives: "
            f"{zero_test}"
        )

    return dist


def check_vocab_leakage(df, scp_df):
    header("7. VOCABULARY AND FREQUENCY-TIER LEAKAGE AUDIT")

    all_vocab, all_counts = build_label_vocab(
        df,
        scp_df,
        min_count=10,
    )

    train_df = df[df["strat_fold"] <= 8].copy()

    train_vocab, train_counts = build_label_vocab(
        train_df,
        scp_df,
        min_count=10,
    )

    all_set = set(all_vocab)
    train_set = set(train_vocab)

    only_all = sorted(all_set - train_set)
    only_train = sorted(train_set - all_set)

    print(
        f"All-data vocabulary size:   {len(all_vocab)}"
    )
    print(
        f"Train-only vocabulary size: {len(train_vocab)}"
    )

    print(
        f"\nLabels introduced only because validation/test were included: "
        f"{len(only_all)}"
    )

    if only_all:
        warn(
            "VOCABULARY LEAKAGE DETECTED. These labels are not present "
            "at least 10 times in training but enter the vocabulary "
            "because validation/test data were used:"
        )
        for code in only_all:
            print(
                f"  {code}: "
                f"all={all_counts.get(code, 0)}, "
                f"train={train_counts.get(code, 0)}"
            )
    else:
        ok(
            "No label enters the vocabulary solely because of "
            "validation/test examples."
        )

    all_tiers = assign_frequency_tiers(
        all_vocab,
        all_counts,
    )

    train_tiers = assign_frequency_tiers(
        train_vocab,
        train_counts,
    )

    changed_tiers = []

    for code in sorted(
        all_set & train_set
    ):
        if all_tiers[code] != train_tiers[code]:
            changed_tiers.append(
                (
                    code,
                    all_counts[code],
                    train_counts[code],
                    all_tiers[code],
                    train_tiers[code],
                )
            )

    print(
        f"\nLabels whose frequency tier changes when using "
        f"train-only counts: {len(changed_tiers)}"
    )

    if changed_tiers:
        warn(
            "FREQUENCY-TIER LEAKAGE DETECTED. Current tiers use "
            "full-dataset label counts."
        )
        print(
            "\n  label      all_count  train_count  "
            "current_tier  train_only_tier"
        )
        for item in changed_tiers:
            print(
                f"  {item[0]:8s} "
                f"{item[1]:10d} "
                f"{item[2]:11d} "
                f"{item[3]:13s} "
                f"{item[4]:16s}"
            )
    else:
        ok(
            "No common/medium/rare tier changes between "
            "all-data and train-only counts."
        )

    return {
        "all_vocab": all_vocab,
        "train_vocab": train_vocab,
        "only_all": only_all,
        "changed_tiers": changed_tiers,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--signal-samples",
        type=int,
        default=5,
        help="Number of recordings sampled per split.",
    )

    args = parser.parse_args()

    print("=" * 78)
    print("PTB-XL DATA + PREPROCESSING AUDIT")
    print("READ-ONLY CHECK — NO FILES WILL BE MODIFIED")
    print("=" * 78)

    print(f"\nPTB-XL directory:")
    print(f"  {PTBXL_DIR}")

    if not PTBXL_DIR.exists():
        fail(
            "PTB-XL directory does not exist."
        )
        return

    try:
        df = load_metadata()
        scp_df = load_scp_statements()
    except Exception as exc:
        fail(
            f"Could not load PTB-XL metadata: {exc}"
        )
        return

    split_sizes = check_metadata(df)

    check_patient_overlap(df)

    check_recordings(
        df,
        sample_per_split=args.signal_samples,
    )

    check_preprocessing(
        df,
        sample_count=max(
            9,
            args.signal_samples * 3,
        ),
    )

    label_info = check_labels(
        df,
        scp_df,
    )

    if label_info is not None:
        dist = check_split_label_distribution(
            df,
            label_info["vocab"],
        )

        leakage = check_vocab_leakage(
            df,
            scp_df,
        )

        header("8. OVERALL INTERPRETATION")

        print(
            "The script has checked the current implementation as written."
        )
        print(
            "A PASS means the current code behaves consistently with "
            "its stated implementation."
        )
        print(
            "A WARN/FAIL may indicate a methodological issue even if "
            "training itself completed successfully."
        )

        if leakage["only_all"] or leakage["changed_tiers"]:
            print(
                "\nIMPORTANT:"
            )
            print(
                "The current code constructs the label vocabulary and "
                "frequency tiers from the COMPLETE PTB-XL metadata, "
                "including validation and test folds."
            )
            print(
                "This does not leak ECG signal values into training, "
                "but it is evaluation metadata leakage and should be "
                "resolved before final research experiments."
            )
        else:
            print(
                "\nNo vocabulary/tier leakage was detected."
            )

        print(
            "\nNo dataset or project files were modified."
        )


if __name__ == "__main__":
    main()
