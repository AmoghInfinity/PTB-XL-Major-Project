import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data" / "ptbxl"

BOOTSTRAP_REPLICATES = 1000
RANDOM_SEED = 42
COVERAGE_LEVELS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
THRESHOLD = 0.5
EPSILON = 1e-12


def load_predictions(path):
    with np.load(path, allow_pickle=False) as data:
        y_true = np.asarray(data["y_true"], dtype=np.int8)
        y_prob = np.asarray(data["y_prob"], dtype=np.float64)

    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch in {path}: {y_true.shape} vs {y_prob.shape}")

    if not np.isfinite(y_prob).all():
        raise ValueError(f"Non-finite probabilities found in {path}")

    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError(f"Probabilities outside [0, 1] found in {path}")

    return y_true, y_prob


def calculate_selective_metrics(y_true, y_prob):
    predictions = y_prob >= THRESHOLD
    record_risk = np.mean(predictions != y_true, axis=1)

    clipped_prob = np.clip(y_prob, EPSILON, 1.0 - EPSILON)
    entropy = -(
        clipped_prob * np.log(clipped_prob)
        + (1.0 - clipped_prob) * np.log(1.0 - clipped_prob)
    )
    uncertainty = np.mean(entropy, axis=1) / np.log(2.0)

    order = np.argsort(uncertainty, kind="stable")
    sorted_risk = record_risk[order]

    accepted_counts = np.arange(1, len(sorted_risk) + 1)
    coverage = accepted_counts / len(sorted_risk)
    cumulative_risk = np.cumsum(sorted_risk) / accepted_counts

    full_coverage = np.concatenate(([0.0], coverage))
    full_risk = np.concatenate(([0.0], cumulative_risk))
    aurc = float(
        np.sum(
            np.diff(full_coverage)
            * (full_risk[:-1] + full_risk[1:])
            / 2.0
        )
    )

    risks = {}
    for level in COVERAGE_LEVELS:
        count = int(np.ceil(level * len(sorted_risk)))
        risks[str(level)] = float(cumulative_risk[count - 1])

    return {"aurc": aurc, "risk_by_coverage": risks}


def percentile_interval(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {"lower": None, "upper": None}

    lower, upper = np.percentile(values, [2.5, 97.5])
    return {"lower": float(lower), "upper": float(upper)}


def summarize_observed(resnet_metrics, mamba_metrics):
    result = {
        "aurc": {
            "resnet1d": resnet_metrics["aurc"],
            "ecg_mamba": mamba_metrics["aurc"],
            "difference_mamba_minus_resnet": (
                mamba_metrics["aurc"] - resnet_metrics["aurc"]
            ),
        },
        "coverage": {},
    }

    for level in COVERAGE_LEVELS:
        key = str(level)
        resnet_risk = resnet_metrics["risk_by_coverage"][key]
        mamba_risk = mamba_metrics["risk_by_coverage"][key]

        relative_reduction = (
            (resnet_risk - mamba_risk) / resnet_risk
            if resnet_risk > 0
            else None
        )

        result["coverage"][key] = {
            "resnet1d_risk": resnet_risk,
            "ecg_mamba_risk": mamba_risk,
            "risk_difference_mamba_minus_resnet": mamba_risk - resnet_risk,
            "relative_risk_reduction": relative_reduction,
        }

    return result


def main():
    resnet_y, resnet_p = load_predictions(
        RESULTS_DIR / "baseline_test_predictions.npz"
    )
    mamba_y, mamba_p = load_predictions(
        RESULTS_DIR / "ecg_mamba_test_predictions.npz"
    )

    if resnet_y.shape != mamba_y.shape:
        raise ValueError("The two models have different target-array shapes.")

    if not np.array_equal(resnet_y, mamba_y):
        raise ValueError("The two models' test targets are not identical.")

    metadata_path = DATA_DIR / "ptbxl_database.csv"
    metadata = pd.read_csv(metadata_path)

    required_columns = {"patient_id", "strat_fold"}
    if not required_columns.issubset(metadata.columns):
        raise ValueError(
            f"Metadata must contain columns: {sorted(required_columns)}"
        )

    test_metadata = metadata.loc[metadata["strat_fold"] == 10].copy()
    patient_ids = test_metadata["patient_id"].to_numpy()

    if len(patient_ids) != len(resnet_y):
        raise ValueError(
            "Test metadata and predictions have different lengths: "
            f"{len(patient_ids)} vs {len(resnet_y)}. "
            "Check that metadata row order matches the saved test predictions."
        )

    if pd.isna(patient_ids).any():
        raise ValueError("Missing patient IDs found in test metadata.")

    unique_patients = np.unique(patient_ids)
    if len(unique_patients) < 2:
        raise ValueError("Not enough unique patients for cluster bootstrap.")

    observed_resnet = calculate_selective_metrics(resnet_y, resnet_p)
    observed_mamba = calculate_selective_metrics(mamba_y, mamba_p)
    observed = summarize_observed(observed_resnet, observed_mamba)

    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_values = {
        "resnet1d": {"aurc": [], "risk": {str(x): [] for x in COVERAGE_LEVELS}},
        "ecg_mamba": {"aurc": [], "risk": {str(x): [] for x in COVERAGE_LEVELS}},
        "difference": {"aurc": [], "risk": {str(x): [] for x in COVERAGE_LEVELS},
                       "relative_risk_reduction": {str(x): [] for x in COVERAGE_LEVELS}},
    }

    patient_indices = {
        patient: np.flatnonzero(patient_ids == patient)
        for patient in unique_patients
    }

    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_patients = rng.choice(
            unique_patients,
            size=len(unique_patients),
            replace=True,
        )
        indices = np.concatenate(
            [patient_indices[patient] for patient in sampled_patients]
        )

        resnet_metrics = calculate_selective_metrics(
            resnet_y[indices], resnet_p[indices]
        )
        mamba_metrics = calculate_selective_metrics(
            mamba_y[indices], mamba_p[indices]
        )

        bootstrap_values["resnet1d"]["aurc"].append(resnet_metrics["aurc"])
        bootstrap_values["ecg_mamba"]["aurc"].append(mamba_metrics["aurc"])
        bootstrap_values["difference"]["aurc"].append(
            mamba_metrics["aurc"] - resnet_metrics["aurc"]
        )

        for level in COVERAGE_LEVELS:
            key = str(level)
            resnet_risk = resnet_metrics["risk_by_coverage"][key]
            mamba_risk = mamba_metrics["risk_by_coverage"][key]

            bootstrap_values["resnet1d"]["risk"][key].append(resnet_risk)
            bootstrap_values["ecg_mamba"]["risk"][key].append(mamba_risk)
            bootstrap_values["difference"]["risk"][key].append(
                mamba_risk - resnet_risk
            )

            reduction = (
                (resnet_risk - mamba_risk) / resnet_risk
                if resnet_risk > 0
                else np.nan
            )
            bootstrap_values["difference"]["relative_risk_reduction"][key].append(
                reduction
            )

        if (replicate + 1) % 100 == 0:
            print(
                f"Completed {replicate + 1}/{BOOTSTRAP_REPLICATES} "
                "bootstrap replicates"
            )

    confidence_intervals = {
        "aurc": {
            "resnet1d": percentile_interval(bootstrap_values["resnet1d"]["aurc"]),
            "ecg_mamba": percentile_interval(bootstrap_values["ecg_mamba"]["aurc"]),
            "difference_mamba_minus_resnet": percentile_interval(
                bootstrap_values["difference"]["aurc"]
            ),
        },
        "coverage": {},
    }

    for level in COVERAGE_LEVELS:
        key = str(level)
        confidence_intervals["coverage"][key] = {
            "resnet1d_risk": percentile_interval(
                bootstrap_values["resnet1d"]["risk"][key]
            ),
            "ecg_mamba_risk": percentile_interval(
                bootstrap_values["ecg_mamba"]["risk"][key]
            ),
            "risk_difference_mamba_minus_resnet": percentile_interval(
                bootstrap_values["difference"]["risk"][key]
            ),
            "relative_risk_reduction": percentile_interval(
                bootstrap_values["difference"]["relative_risk_reduction"][key]
            ),
        }

    output = {
        "method": "Paired patient-cluster bootstrap with percentile 95% confidence intervals",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "random_seed": RANDOM_SEED,
        "threshold": THRESHOLD,
        "test_recordings": int(len(resnet_y)),
        "unique_test_patients": int(len(unique_patients)),
        "labels": int(resnet_y.shape[1]),
        "coverage_levels": COVERAGE_LEVELS,
        "observed": observed,
        "confidence_intervals_95": confidence_intervals,
    }

    output_path = RESULTS_DIR / "selective_prediction_bootstrap.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\nObserved selective prediction results")
    print(f"ResNet1D AURC: {observed['aurc']['resnet1d']:.6f}")
    print(f"ECG-Mamba AURC: {observed['aurc']['ecg_mamba']:.6f}")
    print(
        "AURC difference (Mamba - ResNet1D): "
        f"{observed['aurc']['difference_mamba_minus_resnet']:.6f}"
    )

    for level in COVERAGE_LEVELS:
        key = str(level)
        row = observed["coverage"][key]
        ci = confidence_intervals["coverage"][key]
        print(
            f"Coverage {level:.0%}: "
            f"ResNet risk={row['resnet1d_risk']:.6f}, "
            f"Mamba risk={row['ecg_mamba_risk']:.6f}, "
            f"difference={row['risk_difference_mamba_minus_resnet']:.6f}, "
            f"difference 95% CI="
            f"[{ci['risk_difference_mamba_minus_resnet']['lower']:.6f}, "
            f"{ci['risk_difference_mamba_minus_resnet']['upper']:.6f}]"
        )

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
