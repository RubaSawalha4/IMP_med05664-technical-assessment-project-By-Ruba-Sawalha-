import json
from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================
# VALIDATION SCRIPT – IMPERIAL ASSESSMENT
# Candidate: Ruba Sawalha
# Purpose:
#   - Validate processed sample packs
#   - Check metadata completeness
#   - Check array integrity and harmonisation
#   - Generate manifest and validation report
# ======================================================


# ------------------------------------------------------
# Project paths
# ------------------------------------------------------
PROJECT_ROOT = Path(".")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------
# Expected sample-pack folders and metadata files
# ------------------------------------------------------
DATASET_CONFIG = {
    "pamap2": {
        "folder": PROCESSED_DIR / "pamap2_sample_pack",
        "metadata": "pamap2_sample_pack_metadata.csv",
        "modality": "HAR"
    },
    "wisdm": {
        "folder": PROCESSED_DIR / "wisdm_sample_pack",
        "metadata": "wisdm_sample_pack_metadata.csv",
        "modality": "HAR"
    },
    "eegmmidb": {
        "folder": PROCESSED_DIR / "eeg_sample_pack",
        "metadata": "eeg_sample_pack_metadata.csv",
        "modality": "EEG"
    },
    "ptbxl": {
        "folder": PROCESSED_DIR / "ptbxl_sample_pack",
        "metadata": "ptbxl_sample_pack_metadata.csv",
        "modality": "ECG"
    },
}


# ------------------------------------------------------
# Required metadata columns from the brief
# ------------------------------------------------------
REQUIRED_METADATA_COLUMNS = [
    "sample_id",
    "dataset_name",
    "modality",
    "subject_or_patient_id",
    "source_file_or_record",
    "split",
    "label_or_event",
    "sampling_rate_hz",
    "n_channels",
    "n_samples",
    "channel_schema",
    "qc_flags",
]


# ------------------------------------------------------
# Function: load_npz_signal
# Purpose:
#   Read one saved .npz sample and return its signal array.
# ------------------------------------------------------
def load_npz_signal(file_path: Path):
    with np.load(file_path) as data:
        signal = data["signal"]
    return signal


# ------------------------------------------------------
# Function: check_metadata_columns
# Purpose:
#   Ensure all required metadata columns exist.
# ------------------------------------------------------
def check_metadata_columns(df: pd.DataFrame):
    missing = [col for col in REQUIRED_METADATA_COLUMNS if col not in df.columns]
    return missing


# ------------------------------------------------------
# Function: validate_array_file
# Purpose:
#   Validate one saved sample array:
#   - dtype float32
#   - 2D shape [C, T]
#   - no NaNs or infs
# ------------------------------------------------------
def validate_array_file(file_path: Path):
    signal = load_npz_signal(file_path)

    result = {
        "file_name": file_path.name,
        "dtype_ok": signal.dtype == np.float32,
        "ndim_ok": signal.ndim == 2,
        "has_nan": bool(np.isnan(signal).any()),
        "has_inf": bool(np.isinf(signal).any()),
        "shape": tuple(signal.shape),
        "n_channels": int(signal.shape[0]) if signal.ndim == 2 else None,
        "n_samples": int(signal.shape[1]) if signal.ndim == 2 else None,
        "file_size_bytes": file_path.stat().st_size,
    }

    result["valid"] = (
        result["dtype_ok"]
        and result["ndim_ok"]
        and not result["has_nan"]
        and not result["has_inf"]
    )

    return result


# ------------------------------------------------------
# Function: validate_dataset_pack
# Purpose:
#   Validate one dataset sample-pack folder and metadata.
# ------------------------------------------------------
def validate_dataset_pack(dataset_key, config):
    folder = config["folder"]
    metadata_path = folder / config["metadata"]

    if not folder.exists():
        raise FileNotFoundError(f"Missing processed folder: {folder}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    df = pd.read_csv(metadata_path)
    npz_files = sorted(folder.glob("*.npz"))

    metadata_missing_cols = check_metadata_columns(df)

    array_results = [validate_array_file(f) for f in npz_files]

    summary = {
        "dataset_key": dataset_key,
        "folder": str(folder),
        "metadata_path": str(metadata_path),
        "n_metadata_rows": len(df),
        "n_npz_files": len(npz_files),
        "missing_metadata_columns": metadata_missing_cols,
        "all_arrays_valid": all(r["valid"] for r in array_results) if array_results else False,
        "any_nan": any(r["has_nan"] for r in array_results) if array_results else False,
        "any_inf": any(r["has_inf"] for r in array_results) if array_results else False,
        "unique_dtypes": sorted(list({str(load_npz_signal(f).dtype) for f in npz_files})) if npz_files else [],
    }

    return df, array_results, summary


# ------------------------------------------------------
# Function: check_har_harmonisation
# Purpose:
#   Compare PAMAP2 and WISDM metadata to ensure same:
#   - sampling rate
#   - channel count
#   - channel schema
#   - expected window sizes
# ------------------------------------------------------
def check_har_harmonisation(pamap_df: pd.DataFrame, wisdm_df: pd.DataFrame):
    def get_unique_sorted(series):
        return sorted(series.dropna().astype(str).unique().tolist())

    pamap_rates = get_unique_sorted(pamap_df["sampling_rate_hz"])
    wisdm_rates = get_unique_sorted(wisdm_df["sampling_rate_hz"])

    pamap_channels = get_unique_sorted(pamap_df["n_channels"])
    wisdm_channels = get_unique_sorted(wisdm_df["n_channels"])

    pamap_schema = get_unique_sorted(pamap_df["channel_schema"])
    wisdm_schema = get_unique_sorted(wisdm_df["channel_schema"])

    pamap_samples = get_unique_sorted(pamap_df["n_samples"])
    wisdm_samples = get_unique_sorted(wisdm_df["n_samples"])

    return {
        "pamap_sampling_rates": pamap_rates,
        "wisdm_sampling_rates": wisdm_rates,
        "same_sampling_rate": pamap_rates == wisdm_rates == ["20"],

        "pamap_n_channels": pamap_channels,
        "wisdm_n_channels": wisdm_channels,
        "same_channel_count": pamap_channels == wisdm_channels == ["6"],

        "pamap_channel_schema": pamap_schema,
        "wisdm_channel_schema": wisdm_schema,
        "same_channel_schema": pamap_schema == wisdm_schema,

        "pamap_n_samples": pamap_samples,
        "wisdm_n_samples": wisdm_samples,
        "same_window_sizes": pamap_samples == wisdm_samples == ["100", "200"],
    }


# ------------------------------------------------------
# Function: check_har_label_handling
# Purpose:
#   Ensure supervised HAR outputs do not contain null class 0
#   in the final saved sample packs.
# ------------------------------------------------------
def check_har_label_handling(df: pd.DataFrame):
    supervised_df = df[df["split"] == "supervised"].copy()

    if supervised_df.empty:
        return {"supervised_rows": 0, "contains_label_0": False}

    labels = supervised_df["label_or_event"].astype(str)
    contains_label_0 = (labels == "0").any()

    return {
        "supervised_rows": len(supervised_df),
        "contains_label_0": bool(contains_label_0),
    }


# ------------------------------------------------------
# Function: check_eeg_metadata
# Purpose:
#   Ensure EEG metadata preserves T1/T2 and run info.
# ------------------------------------------------------
def check_eeg_metadata(df: pd.DataFrame):
    labels = sorted(df["label_or_event"].dropna().astype(str).unique().tolist())
    splits = sorted(df["split"].dropna().astype(str).unique().tolist())

    return {
        "unique_event_labels": labels,
        "contains_only_t1_t2": set(labels).issubset({"T1", "T2"}),
        "recorded_runs": splits,
    }


# ------------------------------------------------------
# Function: check_ptbxl_metadata
# Purpose:
#   Ensure ECG fold/split information is present.
# ------------------------------------------------------
def check_ptbxl_metadata(df: pd.DataFrame):
    splits = sorted(df["split"].dropna().astype(str).unique().tolist())

    return {
        "unique_splits": splits,
        "has_test_split": any(s == "test" for s in splits),
        "has_train_cv_splits": any(s.startswith("train_cv_") for s in splits),
    }


# ------------------------------------------------------
# Function: build_processed_manifest
# Purpose:
#   Build one machine-readable manifest for all processed
#   .npz outputs across datasets.
# ------------------------------------------------------
def build_processed_manifest(all_array_rows):
    manifest_df = pd.DataFrame(all_array_rows)
    manifest_path = REPORTS_DIR / "processed_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    return manifest_path


# ------------------------------------------------------
# Function: write_validation_report
# Purpose:
#   Write a short human-readable validation summary.
# ------------------------------------------------------
def write_validation_report(summaries, har_checks, har_label_checks, eeg_checks, ptbxl_checks):
    report_path = REPORTS_DIR / "validation_report.txt"

    lines = []
    lines.append("Validation Report")
    lines.append("=================")
    lines.append("")

    for dataset_key, summary in summaries.items():
        lines.append(f"Dataset: {dataset_key}")
        lines.append(f"  Metadata rows: {summary['n_metadata_rows']}")
        lines.append(f"  NPZ files: {summary['n_npz_files']}")
        lines.append(f"  Missing metadata columns: {summary['missing_metadata_columns']}")
        lines.append(f"  All arrays valid: {summary['all_arrays_valid']}")
        lines.append(f"  Any NaN values: {summary['any_nan']}")
        lines.append(f"  Any infinite values: {summary['any_inf']}")
        lines.append(f"  Dtypes found: {summary['unique_dtypes']}")
        lines.append("")

    lines.append("HAR Harmonisation")
    lines.append("-----------------")
    for k, v in har_checks.items():
        lines.append(f"{k}: {v}")
    lines.append("")

    lines.append("HAR Label Handling")
    lines.append("------------------")
    for dataset_name, check in har_label_checks.items():
        lines.append(f"{dataset_name}: {check}")
    lines.append("")

    lines.append("EEG Metadata Checks")
    lines.append("-------------------")
    for k, v in eeg_checks.items():
        lines.append(f"{k}: {v}")
    lines.append("")

    lines.append("PTB-XL Metadata Checks")
    lines.append("----------------------")
    for k, v in ptbxl_checks.items():
        lines.append(f"{k}: {v}")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path


# ------------------------------------------------------
# Function: main
# Purpose:
#   Run validation on all processed sample packs.
# ------------------------------------------------------
def main():
    all_dataset_frames = {}
    all_summaries = {}
    all_array_rows = []

    for dataset_key, config in DATASET_CONFIG.items():
        print(f"[INFO] Validating dataset: {dataset_key}")

        df, array_results, summary = validate_dataset_pack(dataset_key, config)

        all_dataset_frames[dataset_key] = df
        all_summaries[dataset_key] = summary

        for row in array_results:
            row["dataset_key"] = dataset_key
            all_array_rows.append(row)

    # HAR harmonisation checks
    har_checks = check_har_harmonisation(
        all_dataset_frames["pamap2"],
        all_dataset_frames["wisdm"]
    )

    har_label_checks = {
        "pamap2": check_har_label_handling(all_dataset_frames["pamap2"]),
        "wisdm": check_har_label_handling(all_dataset_frames["wisdm"]),
    }

    # EEG checks
    eeg_checks = check_eeg_metadata(all_dataset_frames["eegmmidb"])

    # PTB-XL checks
    ptbxl_checks = check_ptbxl_metadata(all_dataset_frames["ptbxl"])

    # Write manifest + report
    manifest_path = build_processed_manifest(all_array_rows)
    report_path = write_validation_report(
        summaries=all_summaries,
        har_checks=har_checks,
        har_label_checks=har_label_checks,
        eeg_checks=eeg_checks,
        ptbxl_checks=ptbxl_checks
    )

    print(f"[OK] Processed manifest written: {manifest_path}")
    print(f"[OK] Validation report written: {report_path}")
    print("[OK] Validation completed successfully.")


if __name__ == "__main__":
    main()
