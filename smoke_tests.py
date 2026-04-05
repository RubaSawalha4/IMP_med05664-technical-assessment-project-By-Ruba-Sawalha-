from pathlib import Path
import numpy as np
import pandas as pd

# ======================================================
# SMOKE TESTS – IMPERIAL ASSESSMENT BONUS
# Candidate: Ruba Sawalha
# Purpose:
#   - Quick format checks
#   - Metadata presence checks
#   - Manifest existence checks
# ======================================================

PROJECT_ROOT = Path(".")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATASET_CONFIG = {
    "pamap2": {
        "folder": PROCESSED_DIR / "pamap2_sample_pack",
        "metadata": "pamap2_sample_pack_metadata.csv",
    },
    "wisdm": {
        "folder": PROCESSED_DIR / "wisdm_sample_pack",
        "metadata": "wisdm_sample_pack_metadata.csv",
    },
    "eegmmidb": {
        "folder": PROCESSED_DIR / "eeg_sample_pack",
        "metadata": "eeg_sample_pack_metadata.csv",
    },
    "ptbxl": {
        "folder": PROCESSED_DIR / "ptbxl_sample_pack",
        "metadata": "ptbxl_sample_pack_metadata.csv",
    },
}

REQUIRED_REPORT_FILES = [
    REPORTS_DIR / "download_manifest.csv",
    REPORTS_DIR / "processed_manifest.csv",
    REPORTS_DIR / "validation_report.txt",
    REPORTS_DIR / "resource_estimate.txt",
]


def test_dataset_folder_exists(dataset_key, folder):
    assert folder.exists(), f"{dataset_key}: missing folder {folder}"


def test_metadata_exists(dataset_key, metadata_path):
    assert metadata_path.exists(), f"{dataset_key}: missing metadata file {metadata_path}"


def test_npz_files_exist(dataset_key, folder):
    npz_files = list(folder.glob("*.npz"))
    assert len(npz_files) > 0, f"{dataset_key}: no .npz files found"
    return npz_files


def test_npz_format(dataset_key, npz_files):
    sample = npz_files[0]
    with np.load(sample) as data:
        assert "signal" in data, f"{dataset_key}: missing 'signal' key in {sample.name}"
        signal = data["signal"]
        assert signal.dtype == np.float32, f"{dataset_key}: signal is not float32"
        assert signal.ndim == 2, f"{dataset_key}: signal is not 2D [C, T]"
        assert not np.isnan(signal).any(), f"{dataset_key}: signal contains NaN"
        assert not np.isinf(signal).any(), f"{dataset_key}: signal contains inf"


def test_metadata_rows_match_files(dataset_key, metadata_path, folder):
    df = pd.read_csv(metadata_path)
    npz_files = list(folder.glob("*.npz"))
    assert len(df) == len(npz_files), (
        f"{dataset_key}: metadata rows ({len(df)}) do not match npz files ({len(npz_files)})"
    )


def test_reports_exist():
    for report_file in REQUIRED_REPORT_FILES:
        assert report_file.exists(), f"Missing report/manifest file: {report_file}"


def run_smoke_tests():
    print("[INFO] Running smoke tests...")

    for dataset_key, config in DATASET_CONFIG.items():
        folder = config["folder"]
        metadata_path = folder / config["metadata"]

        test_dataset_folder_exists(dataset_key, folder)
        test_metadata_exists(dataset_key, metadata_path)

        npz_files = test_npz_files_exist(dataset_key, folder)
        test_npz_format(dataset_key, npz_files)
        test_metadata_rows_match_files(dataset_key, metadata_path, folder)

        print(f"[OK] Smoke tests passed for {dataset_key}")

    test_reports_exist()
    print("[OK] Report and manifest smoke tests passed")
    print("[OK] All smoke tests completed successfully.")


if __name__ == "__main__":
    run_smoke_tests()
