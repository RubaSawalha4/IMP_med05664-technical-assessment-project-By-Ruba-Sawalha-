import os
import csv
import time
import socket
import zipfile
import urllib.request
from datetime import datetime

# ======================================================
# SETUP SCRIPT FOR DATA PIPELINE – IMPERIAL ASSESSMENT
# Candidate: Ruba Sawalha
# Purpose:
#   - Create folder structure
#   - Download required datasets
#   - Generate a machine-readable manifest
#   - Retry failed downloads
#   - Extract downloaded ZIP archives
# ======================================================

# ------------------------------------------------------
# Required folder structure for the project
# ------------------------------------------------------
FOLDERS = [
    "configs",            # configuration files
    "data/raw",           # raw downloaded datasets
    "data/interim",       # intermediate outputs
    "data/processed",     # final processed outputs
    "reports",            # reports and manifests
    "submission_sample"   # representative sample pack for submission
]

# ------------------------------------------------------
# Dataset metadata used for scripted downloads
# ------------------------------------------------------
DATASETS = [
    {
        "name": "PAMAP2",
        "modality": "HAR",
        "status": "mandatory",
        "version": "UCI-231",
        "url": "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
        "output_name": "pamap2.zip"
    },
    {
        "name": "WISDM",
        "modality": "HAR",
        "status": "mandatory",
        "version": "UCI-507",
        "url": "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip",
        "output_name": "wisdm.zip"
    },
    {
        "name": "EEGMMIDB",
        "modality": "EEG",
        "status": "mandatory",
        "version": "1.0.0",
        "url": "https://physionet.org/content/eegmmidb/get-zip/1.0.0/",
        "output_name": "eegmmidb.zip"
    },
    {
        "name": "PTB-XL",
        "modality": "ECG",
        "status": "mandatory",
        "version": "1.0.3",
        "url": "https://physionet.org/content/ptb-xl/get-zip/1.0.3/",
        "output_name": "ptbxl.zip"
    }
]

# ------------------------------------------------------
# Function: create_folders
# Purpose:
#   Create the required folder structure for the project.
#   Safe to re-run because exist_ok=True avoids errors
#   if folders already exist.
# ------------------------------------------------------
def create_folders():
    for folder in FOLDERS:
        os.makedirs(folder, exist_ok=True)
        print(f"[OK] Folder ready: {folder}")

# ------------------------------------------------------
# Function: file_exists_and_nonempty
# Purpose:
#   Check whether a file exists and has non-zero size.
#   Used to validate successful downloads and skip
#   already available files.
# ------------------------------------------------------
def file_exists_and_nonempty(path):
    return os.path.exists(path) and os.path.getsize(path) > 0

# ------------------------------------------------------
# Function: download_file
# Purpose:
#   Download a file from a URL with retry logic.
#   If the file already exists and is non-empty,
#   the function skips downloading it again.
#
# Parameters:
#   url          -> source URL
#   output_path  -> destination path on disk
#   retries      -> number of retry attempts
#   wait_seconds -> pause between retries
#
# Returns:
#   status, file_size, error_message
# ------------------------------------------------------
def download_file(url, output_path, retries=3, wait_seconds=5):
    if file_exists_and_nonempty(output_path):
        print(f"[SKIP] File already exists: {output_path}")
        return "skipped_existing", os.path.getsize(output_path), ""

    last_error = ""

    for attempt in range(1, retries + 1):
        try:
            print(f"[INFO] Downloading ({attempt}/{retries}): {url}")

            socket.setdefaulttimeout(120)
            urllib.request.urlretrieve(url, output_path)

            if not file_exists_and_nonempty(output_path):
                raise RuntimeError(f"Downloaded file is missing or empty: {output_path}")

            print(f"[OK] Saved to: {output_path}")
            return "success", os.path.getsize(output_path), ""

        except Exception as e:
            last_error = str(e)
            print(f"[WARN] Attempt {attempt} failed: {last_error}")

            # Remove broken zero-byte files
            if os.path.exists(output_path) and os.path.getsize(output_path) == 0:
                os.remove(output_path)

            if attempt < retries:
                print(f"[INFO] Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)

    return "failed", 0, last_error

# ------------------------------------------------------
# Function: extract_zip
# Purpose:
#   Extract a ZIP archive into a folder inside data/raw.
#   Skips extraction if the target folder already exists.
#
# Parameters:
#   file_path   -> path to zip file
#   extract_to  -> parent folder for extraction
#
# Returns:
#   extraction status string
# ------------------------------------------------------
def extract_zip(file_path, extract_to):
    try:
        folder_name = os.path.splitext(os.path.basename(file_path))[0]
        target_path = os.path.join(extract_to, folder_name)

        if os.path.exists(target_path):
            print(f"[SKIP] Already extracted: {target_path}")
            return "skipped"

        print(f"[INFO] Extracting: {file_path}")

        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(target_path)

        print(f"[OK] Extracted to: {target_path}")
        return "success"

    except Exception as e:
        print(f"[ERROR] Extraction failed: {file_path} -> {e}")
        return "failed"

# ------------------------------------------------------
# Function: write_manifest
# Purpose:
#   Write a machine-readable CSV manifest describing
#   all download attempts, including source URLs,
#   timestamps, file sizes, and errors.
#
# Parameters:
#   rows -> list of dictionaries to save in CSV format
# ------------------------------------------------------
def write_manifest(rows):
    manifest_path = os.path.join("reports", "download_manifest.csv")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_name",
                "modality",
                "status",
                "version",
                "source_url",
                "download_date",
                "output_file",
                "download_status",
                "file_size_bytes",
                "error_message"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Manifest written: {manifest_path}")

# ------------------------------------------------------
# Function: main
# Purpose:
#   Run the setup pipeline end-to-end:
#   1. Create folders
#   2. Download datasets
#   3. Write manifest
#   4. Stop clearly if mandatory downloads fail
#   5. Extract downloaded ZIP files
# ------------------------------------------------------
def main():
    create_folders()

    manifest_rows = []
    failed_mandatory = []

    # Download phase
    for dataset in DATASETS:
        output_path = os.path.join("data", "raw", dataset["output_name"])

        status, file_size, error_message = download_file(
            dataset["url"], output_path, retries=3, wait_seconds=10
        )

        manifest_rows.append({
            "dataset_name": dataset["name"],
            "modality": dataset["modality"],
            "status": dataset["status"],
            "version": dataset["version"],
            "source_url": dataset["url"],
            "download_date": datetime.now().isoformat(),
            "output_file": output_path,
            "download_status": status,
            "file_size_bytes": file_size,
            "error_message": error_message
        })

        if dataset["status"] == "mandatory" and status == "failed":
            failed_mandatory.append(dataset["name"])

    # Save manifest after downloads
    write_manifest(manifest_rows)

    # Fail clearly if any mandatory dataset was not downloaded
    if failed_mandatory:
        raise SystemExit(
            f"[ERROR] Mandatory dataset download failed: {', '.join(failed_mandatory)}"
        )

    print("\n[INFO] Starting extraction phase...\n")

    # Extraction phase
    raw_dir = os.path.join("data", "raw")
    for dataset in DATASETS:
        zip_path = os.path.join(raw_dir, dataset["output_name"])

        if os.path.exists(zip_path):
            extract_zip(zip_path, raw_dir)
        else:
            print(f"[WARN] Missing file, cannot extract: {zip_path}")

    print("[OK] Setup, download, and extraction completed successfully.")

# ------------------------------------------------------
# Script entry point
# ------------------------------------------------------
if __name__ == "__main__":
    main()