import os
import json
import math
import argparse
from pathlib import Path
import mne
import wfdb
import numpy as np
import pandas as pd
print("SCRIPT STARTED")

# ======================================================
# PREPROCESSING SCRIPT – IMPERIAL ASSESSMENT
# Candidate: Ruba Sawalha
# Purpose:
#   - Parse raw datasets
#   - Harmonise HAR data
#   - Prepare EEG and ECG outputs
#   - Save processed arrays and metadata
# ======================================================


# ------------------------------------------------------
# Project paths
# ------------------------------------------------------
PROJECT_ROOT = Path(".")
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ------------------------------------------------------
# Ensure output folders exist
# ------------------------------------------------------
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------
# HAR configuration
# The brief requires PAMAP2 and WISDM to be harmonised
# into one shared representation at 20 Hz, with:
# - 10-second unlabeled windows for pretraining
# - 5-second labeled windows with 50% overlap for
#   supervised evaluation
# ------------------------------------------------------
HAR_TARGET_HZ = 20
HAR_PRETRAIN_WINDOW_SEC = 10
HAR_PRETRAIN_OVERLAP_SEC = 0

HAR_SUPERVISED_WINDOW_SEC = 5
HAR_SUPERVISED_OVERLAP_SEC = 2.5
# ------------------------------------------------------
# Sample-pack saving limits
# We only save a representative subset for submission.
# ------------------------------------------------------
PAMAP2_MAX_PRETRAIN_SAVE = 50
PAMAP2_MAX_SUPERVISED_SAVE = 50
# Shared 6-channel schema:
# accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
HAR_CHANNEL_SCHEMA = [
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]
# ------------------------------------------------------
# WISDM sample limits (100 total)
# ------------------------------------------------------
WISDM_MAX_PRETRAIN_SAVE = 50
WISDM_MAX_SUPERVISED_SAVE = 50

# ------------------------------------------------------
# Unified HAR label schema
# We will refine this mapping when we inspect the exact
# WISDM labels and PAMAP2 activity IDs together.
# For now, keep the target classes explicit and simple.
# ------------------------------------------------------
UNIFIED_HAR_LABELS = {
    "walking": 0,
    "jogging": 1,
    "ascending_stairs": 2,
    "descending_stairs": 3,
    "sitting": 4,
    "standing": 5,
    "lying": 6,
}

# ------------------------------------------------------
# WISDM label mapping → unified HAR labels
# Keep only activities that can be aligned cleanly with
# the shared HAR schema used for PAMAP2 + WISDM.
# ------------------------------------------------------
WISDM_LABEL_MAP = {
    "A": "walking",
    "B": "jogging",
    "C": "ascending_stairs",   # documented simplification: generic stairs → ascending_stairs
    "D": "sitting",
    "E": "standing"
}


# ------------------------------------------------------
# EEGMMIDB paths and configuration
# ------------------------------------------------------
EEG_BASE_DIR = RAW_DIR / "eegmmidb" / "files"
EEG_RUNS = ["R04", "R08", "R12"]
EEG_TARGET_HZ = 160
EEG_WINDOW_SEC = 4
EEG_WINDOW_SAMPLES = EEG_TARGET_HZ * EEG_WINDOW_SEC

# Keep only T1 and T2 for required motor-imagery windows
EEG_ALLOWED_EVENTS = {"T1", "T2"}

# Save 100 EEG samples total
EEG_MAX_SAVE = 100


# ------------------------------------------------------
# PTB-XL paths
# ------------------------------------------------------
PTBXL_BASE_DIR = RAW_DIR / "ptbxl" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
PTBXL_METADATA_PATH = PTBXL_BASE_DIR / "ptbxl_database.csv"
PTBXL_RECORDS_DIR = PTBXL_BASE_DIR / "records100"
PTBXL_TARGET_HZ = 100

# Save 100 samples
PTBXL_MAX_SAVE = 100
# Read metadata Ptbxl
def load_ptbxl_metadata():
    df = pd.read_csv(PTBXL_METADATA_PATH)

    # Convert scp_codes string → dict
    df["scp_codes"] = df["scp_codes"].apply(eval)

    return df
    
def read_ptbxl_signal(record_path):
    signal, meta = wfdb.rdsamp(str(record_path))
    return signal.astype(np.float32), meta
# ------------------------------------------------------
# Function: resample_array
# Purpose:
#   Resample a multichannel time-series from one sampling
#   rate to another using linear interpolation.
#
# Input shape:
#   data -> [T, C]
#
# Output shape:
#   [new_T, C]
# ------------------------------------------------------
def resample_array(data: np.ndarray, orig_hz: float, target_hz: float) -> np.ndarray:
    if orig_hz == target_hz:
        return data.astype(np.float32)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D array [T, C], got shape {data.shape}")

    n_samples, n_channels = data.shape
    duration_sec = n_samples / orig_hz
    new_n_samples = int(round(duration_sec * target_hz))

    old_time = np.linspace(0, duration_sec, n_samples, endpoint=False)
    new_time = np.linspace(0, duration_sec, new_n_samples, endpoint=False)

    resampled = np.empty((new_n_samples, n_channels), dtype=np.float32)

    for ch in range(n_channels):
        resampled[:, ch] = np.interp(new_time, old_time, data[:, ch])

    return resampled


# ------------------------------------------------------
# Function: create_windows
# Purpose:
#   Split a multichannel time-series into fixed windows.
#
# Input:
#   data -> [T, C]
#
# Output:
#   windows -> [N, C, W]
# ------------------------------------------------------
def create_windows(
    data: np.ndarray,
    sampling_rate_hz: int,
    window_sec: float,
    overlap_sec: float,
) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array [T, C], got shape {data.shape}")

    window_size = int(window_sec * sampling_rate_hz)
    step_size = int((window_sec - overlap_sec) * sampling_rate_hz)

    if window_size <= 0 or step_size <= 0:
        raise ValueError("Window size and step size must be positive")

    windows = []
    n_samples = data.shape[0]

    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        window = data[start:end]          # [W, C]
        window = window.T                 # [C, W]
        windows.append(window.astype(np.float32))

    if not windows:
        return np.empty((0, data.shape[1], window_size), dtype=np.float32)

    return np.stack(windows, axis=0)


# ------------------------------------------------------
# Function: majority_label
# Purpose:
#   Assign one label to a supervised HAR window using the
#   majority class inside that window.
#
# Input:
#   labels -> 1D label array aligned to samples
# ------------------------------------------------------
def majority_label(labels: np.ndarray):
    if len(labels) == 0:
        return None

    values, counts = np.unique(labels, return_counts=True)
    return values[np.argmax(counts)]


# ------------------------------------------------------
# Function: zscore_normalize
# Purpose:
#   Apply per-channel z-score normalization.
#
# Input:
#   data -> [T, C]
# ------------------------------------------------------
def zscore_normalize(data: np.ndarray) -> np.ndarray:
    if data.size == 0:
        return data.astype(np.float32)

    mean = np.nanmean(data, axis=0, keepdims=True)
    std = np.nanstd(data, axis=0, keepdims=True)
    std[std == 0] = 1.0

    normalized = (data - mean) / std
    return normalized.astype(np.float32)

# ------------------------------------------------------
# Function: make_metadata_row
# Purpose:
#   Build one metadata record per processed sample.
# ------------------------------------------------------
def make_metadata_row(
    sample_id,
    dataset_name,
    modality,
    subject_or_patient_id,
    source_file_or_record,
    split,
    label_or_event,
    sampling_rate_hz,
    n_channels,
    n_samples,
    channel_schema,
    qc_flags=""
):
    return {
        "sample_id": sample_id,
        "dataset_name": dataset_name,
        "modality": modality,
        "subject_or_patient_id": subject_or_patient_id,
        "source_file_or_record": source_file_or_record,
        "split": split,
        "label_or_event": label_or_event,
        "sampling_rate_hz": sampling_rate_hz,
        "n_channels": n_channels,
        "n_samples": n_samples,
        "channel_schema": json.dumps(channel_schema),
        "qc_flags": qc_flags,
    }


# ------------------------------------------------------
# Function: save_npz_sample
# Purpose:
#   Save one processed sample/window as compressed NumPy.
# ------------------------------------------------------
def save_npz_sample(output_path: Path, signal_array: np.ndarray):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, signal=signal_array.astype(np.float32))

# ------------------------------------------------------
# PAMAP2 paths and column selection
# We use the Protocol folder and start with one subject.
# ------------------------------------------------------
PAMAP2_PROTOCOL_DIR = RAW_DIR / "pamap2" / "PAMAP2_Dataset" / "Protocol"

# PAMAP2 column indices based on the dataset layout.
# We will use wrist IMU only:
# - activity_id
# - wrist accel x,y,z
# - wrist gyro x,y,z
#
# Note:
# PAMAP2 files are space-separated with no header.
# ------------------------------------------------------
PAMAP2_ACTIVITY_COL = 1

# Wrist IMU columns
# These are the standard wrist IMU positions in PAMAP2.
PAMAP2_WRIST_ACCEL_COLS = [22, 23, 24]
PAMAP2_WRIST_GYRO_COLS = [28, 29, 30]

PAMAP2_SELECTED_COLS = (
    [PAMAP2_ACTIVITY_COL] +
    PAMAP2_WRIST_ACCEL_COLS +
    PAMAP2_WRIST_GYRO_COLS
)

PAMAP2_ORIG_HZ = 100


# ------------------------------------------------------
# WISDM paths
# ------------------------------------------------------
WISDM_BASE_DIR = RAW_DIR / "wisdm" / "wisdm-dataset" / "raw" / "watch"
WISDM_ACCEL_DIR = WISDM_BASE_DIR / "accel"
WISDM_GYRO_DIR = WISDM_BASE_DIR / "gyro"

# ------------------------------------------------------
# Function: read_pamap2_subject
# Purpose:
#   Read one PAMAP2 subject file and return:
#   - signal array [T, 6]
#   - activity labels [T]
#
# Processing:
#   - load only needed columns
#   - drop rows with missing values
#   - cast to numeric
# ------------------------------------------------------
def read_pamap2_subject(file_path: Path):
    print(f"[INFO] Reading PAMAP2 file: {file_path.name}")

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        usecols=PAMAP2_SELECTED_COLS,
        engine="python"
    )

    df.columns = [
        "activity_id",
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z"
    ]

    # Drop rows with missing values in selected channels
    df = df.dropna().reset_index(drop=True)

    labels = df["activity_id"].to_numpy()
    signals = df[HAR_CHANNEL_SCHEMA].to_numpy(dtype=np.float32)

    return signals, labels


# ------------------------------------------------------
# Function: create_label_windows
# Purpose:
#   Create one label per supervised HAR window using
#   majority vote over the samples in the window.
# ------------------------------------------------------
def create_label_windows(
    labels: np.ndarray,
    sampling_rate_hz: int,
    window_sec: float,
    overlap_sec: float
):
    window_size = int(window_sec * sampling_rate_hz)
    step_size = int((window_sec - overlap_sec) * sampling_rate_hz)

    out_labels = []

    for start in range(0, len(labels) - window_size + 1, step_size):
        end = start + window_size
        window_labels = labels[start:end]
        out_labels.append(majority_label(window_labels))

    return out_labels


# ------------------------------------------------------
# Function: process_pamap2_subject
# Purpose:
#   Process one PAMAP2 subject into:
#   - pretraining windows
#   - supervised windows
#   - metadata rows
#
# Notes:
#   - class 0 is treated as transient/null and excluded
#     from supervised outputs
#   - signals are normalized after resampling
# ------------------------------------------------------
def process_pamap2_subject(file_path: Path):
    subject_id = file_path.stem  # e.g. subject101

    signals, labels = read_pamap2_subject(file_path)

    # Resample signals to common HAR target rate
    signals_resampled = resample_array(signals, PAMAP2_ORIG_HZ, HAR_TARGET_HZ)

    # Resample labels by nearest-neighbour style indexing
    old_idx = np.arange(len(labels))
    new_idx = np.linspace(0, len(labels) - 1, len(signals_resampled))
    labels_resampled = labels[np.round(new_idx).astype(int)]

    # Normalize signals channel-wise
    signals_resampled = zscore_normalize(signals_resampled)

    # -------------------------
    # Pretraining windows
    # -------------------------
    pretrain_windows = create_windows(
        signals_resampled,
        HAR_TARGET_HZ,
        HAR_PRETRAIN_WINDOW_SEC,
        HAR_PRETRAIN_OVERLAP_SEC
    )

    # -------------------------
    # Supervised windows
    # -------------------------
    supervised_windows = create_windows(
        signals_resampled,
        HAR_TARGET_HZ,
        HAR_SUPERVISED_WINDOW_SEC,
        HAR_SUPERVISED_OVERLAP_SEC
    )

    supervised_labels = create_label_windows(
        labels_resampled,
        HAR_TARGET_HZ,
        HAR_SUPERVISED_WINDOW_SEC,
        HAR_SUPERVISED_OVERLAP_SEC
    )

    # Exclude transient / null PAMAP2 class 0 for supervised set
    filtered_windows = []
    filtered_labels = []

    for w, lbl in zip(supervised_windows, supervised_labels):
        if lbl == 0:
            continue
        filtered_windows.append(w)
        filtered_labels.append(lbl)

    if filtered_windows:
        supervised_windows = np.stack(filtered_windows, axis=0)
    else:
        supervised_windows = np.empty(
            (0, len(HAR_CHANNEL_SCHEMA), int(HAR_SUPERVISED_WINDOW_SEC * HAR_TARGET_HZ)),
            dtype=np.float32
        )

    return {
        "subject_id": subject_id,
        "pretrain_windows": pretrain_windows,
        "supervised_windows": supervised_windows,
        "supervised_labels": filtered_labels,
    }

# ------------------------------------------------------
# Function: process_all_pamap2_subjects
# Purpose:
#   Process all PAMAP2 Protocol subject files and return
#   a list of processed subject dictionaries.
# ------------------------------------------------------
def process_all_pamap2_subjects():
    subject_files = sorted(PAMAP2_PROTOCOL_DIR.glob("subject*.dat"))

    if not subject_files:
        raise FileNotFoundError(f"No PAMAP2 subject files found in: {PAMAP2_PROTOCOL_DIR}")

    all_processed = []

    for file_path in subject_files:
        print(f"\n[INFO] Processing PAMAP2 subject: {file_path.name}")
        processed = process_pamap2_subject(file_path)

        print(f"[INFO] {processed['subject_id']} pretraining windows: {processed['pretrain_windows'].shape}")
        print(f"[INFO] {processed['subject_id']} supervised windows: {processed['supervised_windows'].shape}")
        print(f"[INFO] {processed['subject_id']} supervised labels: {len(processed['supervised_labels'])}")

        all_processed.append(processed)

    return all_processed
# ------------------------------------------------------
# Function: save_pamap2_preview_outputs
# Purpose:
#   Save a small preview from one PAMAP2 subject so we
#   can verify shapes and metadata before scaling up.
# ------------------------------------------------------
# ------------------------------------------------------
# Function: save_pamap2_sample_pack
# Purpose:
#   Save a representative PAMAP2 sample pack across all
#   subjects, limited to a fixed number of unlabeled and
#   supervised windows.
#
# Returns:
#   metadata_df
# ------------------------------------------------------
def save_pamap2_sample_pack(all_processed_subjects):
    output_dir = PROCESSED_DIR / "pamap2_sample_pack"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []

    saved_pretrain = 0
    saved_supervised = 0

    for processed in all_processed_subjects:
        subject_id = processed["subject_id"]

        # Save pretraining windows
        for i, window in enumerate(processed["pretrain_windows"]):
            if saved_pretrain >= PAMAP2_MAX_PRETRAIN_SAVE:
                break

            sample_id = f"{subject_id}_pretrain_{saved_pretrain:03d}"
            output_path = output_dir / f"{sample_id}.npz"
            save_npz_sample(output_path, window)

            metadata_rows.append(
                make_metadata_row(
                    sample_id=sample_id,
                    dataset_name="PAMAP2",
                    modality="HAR",
                    subject_or_patient_id=subject_id,
                    source_file_or_record=subject_id,
                    split="unsupervised",
                    label_or_event="",
                    sampling_rate_hz=HAR_TARGET_HZ,
                    n_channels=window.shape[0],
                    n_samples=window.shape[1],
                    channel_schema=HAR_CHANNEL_SCHEMA,
                    qc_flags=""
                )
            )

            saved_pretrain += 1

        # Save supervised windows
        for i, (window, lbl) in enumerate(zip(processed["supervised_windows"], processed["supervised_labels"])):
            if saved_supervised >= PAMAP2_MAX_SUPERVISED_SAVE:
                break

            sample_id = f"{subject_id}_supervised_{saved_supervised:03d}"
            output_path = output_dir / f"{sample_id}.npz"
            save_npz_sample(output_path, window)

            metadata_rows.append(
                make_metadata_row(
                    sample_id=sample_id,
                    dataset_name="PAMAP2",
                    modality="HAR",
                    subject_or_patient_id=subject_id,
                    source_file_or_record=subject_id,
                    split="supervised",
                    label_or_event=int(lbl),
                    sampling_rate_hz=HAR_TARGET_HZ,
                    n_channels=window.shape[0],
                    n_samples=window.shape[1],
                    channel_schema=HAR_CHANNEL_SCHEMA,
                    qc_flags=""
                )
            )

            saved_supervised += 1

        if (
            saved_pretrain >= PAMAP2_MAX_PRETRAIN_SAVE
            and saved_supervised >= PAMAP2_MAX_SUPERVISED_SAVE
        ):
            break

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = output_dir / "pamap2_sample_pack_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    print(f"[OK] Saved PAMAP2 sample pack to: {output_dir}")
    print(f"[OK] PAMAP2 metadata written: {metadata_path}")
    print(f"[INFO] Saved pretraining windows: {saved_pretrain}")
    print(f"[INFO] Saved supervised windows: {saved_supervised}")

    return metadata_df

# ------------------------------------------------------
# ------------------------------------------------------
# Function: read_wisdm_file
# Purpose:
#   Read a WISDM sensor file (accel or gyro)
#   Format: user, activity, timestamp, x, y, z
# ------------------------------------------------------
# ------------------------------------------------------
# Function: read_wisdm_file
# Purpose:
#   Read a WISDM sensor file (accel or gyro)
#   Format: user, activity, timestamp, x, y, z
# ------------------------------------------------------
def read_wisdm_file(file_path: Path):
    df = pd.read_csv(
        file_path,
        header=None,
        names=["user", "activity", "timestamp", "x", "y", "z"],
        sep=",",
        engine="python"
    )

    # Clean activity labels
    df["activity"] = (
        df["activity"]
        .astype(str)
        .str.replace(";", "", regex=False)
        .str.strip()
        .str.title()
    )

    # Clean numeric columns
    for col in ["x", "y", "z"]:
        df[col] = df[col].astype(str).str.replace(";", "", regex=False).str.strip()

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["z"] = pd.to_numeric(df["z"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    return df
# ------------------------------------------------------
# Function: merge_wisdm_signals
# Purpose:
#   Combine accel and gyro into 6-channel signal
# ------------------------------------------------------
def merge_wisdm_signals(accel_df, gyro_df):
    min_len = min(len(accel_df), len(gyro_df))

    accel_df = accel_df.iloc[:min_len]
    gyro_df = gyro_df.iloc[:min_len]

    signals = np.stack([
        accel_df["x"].values,
        accel_df["y"].values,
        accel_df["z"].values,
        gyro_df["x"].values,
        gyro_df["y"].values,
        gyro_df["z"].values,
    ], axis=1)

    labels = accel_df["activity"].values

    return signals.astype(np.float32), labels
# ------------------------------------------------------
# Function: process_wisdm_pair
# Purpose:
#   Process one paired WISDM accel/gyro recording
# ------------------------------------------------------
def process_wisdm_pair(accel_path: Path, gyro_path: Path):
    file_id = accel_path.name.split("_")[1]  # 1600 etc

    accel_df = read_wisdm_file(accel_path)
    gyro_df = read_wisdm_file(gyro_path)

    signals, labels_raw = merge_wisdm_signals(accel_df, gyro_df)
    
    # Map labels
    labels = []
    for lbl in labels_raw:
        mapped = WISDM_LABEL_MAP.get(lbl)
        if mapped is None:
            labels.append(None)
        else:
            labels.append(UNIFIED_HAR_LABELS[mapped])

    labels = np.array(labels)
    

    # Remove invalid labels
    valid_mask = pd.notna(labels)
    signals = signals[valid_mask]
    labels = labels[valid_mask].astype(int)

    # Normalize
    signals = zscore_normalize(signals)

    # Windows
    pretrain_windows = create_windows(
        signals, HAR_TARGET_HZ,
        HAR_PRETRAIN_WINDOW_SEC,
        HAR_PRETRAIN_OVERLAP_SEC
    )

    supervised_windows = create_windows(
        signals, HAR_TARGET_HZ,
        HAR_SUPERVISED_WINDOW_SEC,
        HAR_SUPERVISED_OVERLAP_SEC
    )

    supervised_labels = create_label_windows(
        labels,
        HAR_TARGET_HZ,
        HAR_SUPERVISED_WINDOW_SEC,
        HAR_SUPERVISED_OVERLAP_SEC
    )

    return {
        "id": file_id,
        "pretrain_windows": pretrain_windows,
        "supervised_windows": supervised_windows,
        "supervised_labels": supervised_labels
    }
# ------------------------------------------------------
# Function: process_all_wisdm_pairs
# Purpose:
#   Match accel and gyro files and process all pairs
# ------------------------------------------------------
def process_all_wisdm_pairs():
    accel_files = sorted(WISDM_ACCEL_DIR.glob("data_*_accel_watch.txt"))

    if not accel_files:
        raise FileNotFoundError("No WISDM accel files found")

    all_processed = []

    for accel_path in accel_files:
        file_id = accel_path.name.split("_")[1]
        gyro_path = WISDM_GYRO_DIR / f"data_{file_id}_gyro_watch.txt"

        if not gyro_path.exists():
            print(f"[WARN] Missing gyro file for {file_id}, skipping")
            continue

        print(f"\n[INFO] Processing WISDM ID: {file_id}")

        processed = process_wisdm_pair(accel_path, gyro_path)

        print(f"[INFO] {file_id} pretraining windows: {processed['pretrain_windows'].shape}")
        print(f"[INFO] {file_id} supervised windows: {processed['supervised_windows'].shape}")
        print(f"[INFO] {file_id} supervised labels: {len(processed['supervised_labels'])}")

        all_processed.append(processed)

    return all_processed
# ------------------------------------------------------
# Function: save_wisdm_sample_pack
# Purpose:
#   Save representative WISDM samples (100 total)
# ------------------------------------------------------
def save_wisdm_sample_pack(all_processed):
    output_dir = PROCESSED_DIR / "wisdm_sample_pack"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []

    saved_pretrain = 0
    saved_supervised = 0

    for processed in all_processed:
        file_id = processed["id"]

        # Pretraining
        for window in processed["pretrain_windows"]:
            if saved_pretrain >= WISDM_MAX_PRETRAIN_SAVE:
                break

            sample_id = f"{file_id}_pretrain_{saved_pretrain:03d}"
            output_path = output_dir / f"{sample_id}.npz"
            save_npz_sample(output_path, window)

            metadata_rows.append(
                make_metadata_row(
                    sample_id=sample_id,
                    dataset_name="WISDM",
                    modality="HAR",
                    subject_or_patient_id=file_id,
                    source_file_or_record=file_id,
                    split="unsupervised",
                    label_or_event="",
                    sampling_rate_hz=HAR_TARGET_HZ,
                    n_channels=window.shape[0],
                    n_samples=window.shape[1],
                    channel_schema=HAR_CHANNEL_SCHEMA,
                    qc_flags=""
                )
            )

            saved_pretrain += 1

        # Supervised
        for window, lbl in zip(processed["supervised_windows"], processed["supervised_labels"]):
            if saved_supervised >= WISDM_MAX_SUPERVISED_SAVE:
                break

            sample_id = f"{file_id}_supervised_{saved_supervised:03d}"
            output_path = output_dir / f"{sample_id}.npz"
            save_npz_sample(output_path, window)

            metadata_rows.append(
                make_metadata_row(
                    sample_id=sample_id,
                    dataset_name="WISDM",
                    modality="HAR",
                    subject_or_patient_id=file_id,
                    source_file_or_record=file_id,
                    split="supervised",
                    label_or_event=int(lbl),
                    sampling_rate_hz=HAR_TARGET_HZ,
                    n_channels=window.shape[0],
                    n_samples=window.shape[1],
                    channel_schema=HAR_CHANNEL_SCHEMA,
                    qc_flags=""
                )
            )

            saved_supervised += 1

        if saved_pretrain >= WISDM_MAX_PRETRAIN_SAVE and saved_supervised >= WISDM_MAX_SUPERVISED_SAVE:
            break

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = output_dir / "wisdm_sample_pack_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    print(f"[OK] Saved WISDM sample pack to: {output_dir}")
    print(f"[INFO] Saved pretraining windows: {saved_pretrain}")
    print(f"[INFO] Saved supervised windows: {saved_supervised}")

    return metadata_df

# ------------------------------------------------------
# Function: parse_run_id
# Purpose:
#   Extract run ID like R04 from filename S001R04.edf
# ------------------------------------------------------
def parse_run_id(file_path: Path):
    stem = file_path.stem  # S001R04
    return stem[-3:]


# ------------------------------------------------------
# Function: parse_subject_id
# Purpose:
#   Extract subject ID like S001 from filename
# ------------------------------------------------------
def parse_subject_id(file_path: Path):
    stem = file_path.stem  # S001R04
    return stem[:4]


# ------------------------------------------------------
# Function: read_eeg_edf
# Purpose:
#   Read one EDF file with MNE and return:
#   - signal array [T, C]
#   - sampling rate
#   - annotations
#   - channel names
# ------------------------------------------------------
def read_eeg_edf(file_path: Path):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose="ERROR")

    sampling_rate = int(raw.info["sfreq"])
    channel_names = raw.ch_names

    # Light preprocessing: channel-wise standardization only
    data = raw.get_data().T.astype(np.float32)   # [T, C]
    data = zscore_normalize(data)

    annotations = raw.annotations
    return data, sampling_rate, annotations, channel_names


# ------------------------------------------------------
# Function: extract_eeg_event_windows
# Purpose:
#   Create fixed 4-second windows from T1/T2 onset.
#   Preserve event code and timing metadata.
# ------------------------------------------------------
def extract_eeg_event_windows(
    data: np.ndarray,
    sampling_rate: int,
    annotations,
    subject_id: str,
    run_id: str,
    source_file: str,
):
    windows = []
    metadata_rows = []

    for i in range(len(annotations)):
        event_desc = str(annotations.description[i]).strip()
        onset_sec = float(annotations.onset[i])

        if event_desc not in EEG_ALLOWED_EVENTS:
            continue

        start_idx = int(round(onset_sec * sampling_rate))
        end_idx = start_idx + EEG_WINDOW_SAMPLES

        # Skip incomplete windows
        if end_idx > len(data):
            continue

        window = data[start_idx:end_idx].T.astype(np.float32)  # [C, T]
        sample_id = f"{subject_id}_{run_id}_{event_desc}_{start_idx}"

        windows.append((sample_id, window))

        metadata_rows.append(
            make_metadata_row(
                sample_id=sample_id,
                dataset_name="EEGMMIDB",
                modality="EEG",
                subject_or_patient_id=subject_id,
                source_file_or_record=source_file,
                split=run_id,
                label_or_event=event_desc,
                sampling_rate_hz=sampling_rate,
                n_channels=window.shape[0],
                n_samples=window.shape[1],
                channel_schema=[],
                qc_flags=f"onset_sec={onset_sec}"
            )
        )

    return windows, metadata_rows


# ------------------------------------------------------
# Function: process_all_eeg_subjects
# Purpose:
#   Loop through subject folders and process runs 4, 8, 12
# ------------------------------------------------------
def process_all_eeg_subjects():
    subject_dirs = sorted([p for p in EEG_BASE_DIR.iterdir() if p.is_dir() and p.name.startswith("S")])

    if not subject_dirs:
        raise FileNotFoundError(f"No EEG subject folders found in: {EEG_BASE_DIR}")

    all_windows = []
    all_metadata = []

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name

        for edf_path in sorted(subject_dir.glob("*.edf")):
            run_id = parse_run_id(edf_path)

            if run_id not in EEG_RUNS:
                continue

            print(f"\n[INFO] Processing EEG file: {edf_path.name}")

            data, sampling_rate, annotations, channel_names = read_eeg_edf(edf_path)

            # Keep native 160 Hz as required by our plan
            if sampling_rate != EEG_TARGET_HZ:
                print(f"[WARN] Unexpected EEG sampling rate in {edf_path.name}: {sampling_rate}")

            windows, metadata_rows = extract_eeg_event_windows(
                data=data,
                sampling_rate=sampling_rate,
                annotations=annotations,
                subject_id=subject_id,
                run_id=run_id,
                source_file=edf_path.name
            )

            print(f"[INFO] Extracted EEG windows: {len(windows)}")

            # Add channel names to metadata rows
            for row in metadata_rows:
                row["channel_schema"] = json.dumps(channel_names)

            all_windows.extend(windows)
            all_metadata.extend(metadata_rows)

    return all_windows, all_metadata


# ------------------------------------------------------
# Function: save_eeg_sample_pack
# Purpose:
#   Save 100 representative EEG windows + metadata
# ------------------------------------------------------
def save_eeg_sample_pack(all_windows, all_metadata):
    output_dir = PROCESSED_DIR / "eeg_sample_pack"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    kept_rows = []

    metadata_lookup = {row["sample_id"]: row for row in all_metadata}

    for sample_id, window in all_windows:
        if saved >= EEG_MAX_SAVE:
            break

        output_path = output_dir / f"{sample_id}.npz"
        save_npz_sample(output_path, window)

        kept_rows.append(metadata_lookup[sample_id])
        saved += 1

    metadata_df = pd.DataFrame(kept_rows)
    metadata_path = output_dir / "eeg_sample_pack_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    print(f"[OK] Saved EEG sample pack to: {output_dir}")
    print(f"[INFO] Saved EEG windows: {saved}")

    return metadata_df
    
###------- process ptbxl -----------------
def process_ptbxl():
    df = load_ptbxl_metadata()

    print(f"[INFO] PTB-XL metadata rows: {len(df)}")
    print(f"[INFO] Example record path: {df.iloc[0]['filename_lr']}")

    samples = []
    metadata_rows = []

    # Process only enough records to build the required sample pack
    for row in df.itertuples(index=False):
        record_name = row.filename_lr   # e.g. records100/00000/00001_lr
        record_path = PTBXL_BASE_DIR / record_name

        try:
            signal, meta = read_ptbxl_signal(record_path)
        except Exception as e:
            print(f"[WARN] Failed to read PTB-XL record {record_name}: {e}")
            continue

        # Convert [T, C] -> [C, T]
        signal = signal.T.astype(np.float32)

        # Normalize channel-wise
        signal = zscore_normalize(signal.T).T

        sample_id = str(row.ecg_id)

        samples.append((sample_id, signal))

        # Use strat_fold to preserve fold information
        # Example convention:
        # fold 10 -> test
        # folds 1-9 -> train_cv_<fold>
        if row.strat_fold == 10:
            split_name = "test"
        else:
            split_name = f"train_cv_{row.strat_fold}"

        metadata_rows.append(
            make_metadata_row(
                sample_id=sample_id,
                dataset_name="PTB-XL",
                modality="ECG",
                subject_or_patient_id=row.patient_id,
                source_file_or_record=record_name,
                split=split_name,
                label_or_event=str(row.scp_codes),
                sampling_rate_hz=PTBXL_TARGET_HZ,
                n_channels=signal.shape[0],
                n_samples=signal.shape[1],
                channel_schema=[],
                qc_flags=""
            )
        )

        if len(samples) % 10 == 0:
            print(f"[INFO] Collected PTB-XL samples: {len(samples)}")

        if len(samples) >= PTBXL_MAX_SAVE:
            break

    return samples, metadata_rows
#------------- save sample pack ptbxl
def save_ptbxl_sample_pack(samples, metadata_rows):
    output_dir = PROCESSED_DIR / "ptbxl_sample_pack"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    kept_rows = []

    metadata_lookup = {row["sample_id"]: row for row in metadata_rows}

    for sample_id, signal in samples:
        if saved >= PTBXL_MAX_SAVE:
            break

        output_path = output_dir / f"{sample_id}.npz"
        save_npz_sample(output_path, signal)

        kept_rows.append(metadata_lookup[sample_id])
        saved += 1

    metadata_df = pd.DataFrame(kept_rows)
    metadata_df.to_csv(output_dir / "ptbxl_sample_pack_metadata.csv", index=False)

    print(f"[OK] Saved PTB-XL sample pack")
    print(f"[INFO] Saved samples: {saved}")

    return metadata_df
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["pamap2", "wisdm", "eegmmidb", "ptbxl"])
    args = parser.parse_args()

    print(f"[INFO] Selected dataset: {args.dataset}")

    if args.dataset == "pamap2":
        all_processed = process_all_pamap2_subjects()
        metadata_df = save_pamap2_sample_pack(all_processed)

        print(f"[OK] PAMAP2 full-subject processing completed.")
        print(f"[INFO] Total saved PAMAP2 samples: {len(metadata_df)}")
    elif args.dataset == "wisdm":
        all_processed = process_all_wisdm_pairs()
        metadata_df = save_wisdm_sample_pack(all_processed)

        print(f"[OK] WISDM processing completed.")
        print(f"[INFO] Total saved WISDM samples: {len(metadata_df)}")
    elif args.dataset == "eegmmidb":
        all_windows, all_metadata = process_all_eeg_subjects()
        metadata_df = save_eeg_sample_pack(all_windows, all_metadata)

        print(f"[OK] EEGMMIDB processing completed.")
        print(f"[INFO] Total saved EEG samples: {len(metadata_df)}")
    elif args.dataset == "ptbxl":
        samples, metadata = process_ptbxl()
        metadata_df = save_ptbxl_sample_pack(samples, metadata)

        print(f"[OK] PTB-XL processing completed.")
        print(f"[INFO] Total saved PTB-XL samples: {len(metadata_df)}")
    else:
        print(f"[INFO] {args.dataset} pipeline not added yet.")
# ------------------------------------------------------
# Script entry point
# ------------------------------------------------------
if __name__ == "__main__":
    main()