#Multi-Modal Biomedical Data Preprocessing Pipeline (MED05664 Technical Assessment_Candidate:RubaSawalha)

## Overview

This project implements a complete data preprocessing pipeline for multiple biomedical datasets:

* **HAR (Human Activity Recognition):** PAMAP2, WISDM
* **EEG:** EEGMMIDB
* **ECG:** PTB-XL

The pipeline performs:

* dataset download and organisation
* signal ingestion and preprocessing
* harmonisation across datasets
* window generation
* metadata creation
* validation and integrity checks

The output is a set of **fixed-shape float32 arrays with structured metadata**, suitable for downstream machine learning or self-supervised learning.

---

## Project Structure

```
pipeline_project/
│
├── setup_data.py
├── preprocess.py
├── validate_outputs.py
├── smoke_tests.py
├── README.md
│
├── configs/
│   ├── har_config.json
├── reports/
│   ├── preprocessing_plan.pdf
│   ├── validation_report.txt
│   ├── resource_estimate.txt
│   ├── self_supervised_note.txt
│   ├── download_manifest.csv
│   └── processed_manifest.csv
│
├── submission_sample/
│   ├── pamap2/
│   ├── wisdm/
│   ├── eegmmidb/
│   └── ptbxl/
```

---

## Setup Instructions

### 1. Install dependencies

```
pip install numpy pandas mne wfdb
```

---

### 2. Download datasets and create folder structure

```
python setup_data.py
```

This will:

* create required directories
* download all datasets
* generate a download manifest

---

### 3. Run preprocessing

Run each dataset independently:

```
python preprocess.py --dataset pamap2
python preprocess.py --dataset wisdm
python preprocess.py --dataset eegmmidb
python preprocess.py --dataset ptbxl
```

---

### 4. Run validation

```
python validate_outputs.py
```

This generates:

* `processed_manifest.csv`
* `validation_report.txt`

---

### 5. Run smoke tests (bonus)

```
python smoke_tests.py
```

---

## Output Format

* Signals stored as `.npz` files containing:

  * `signal`: float32 array of shape `[C, T]`
* Each dataset includes:

  * 100 representative samples
  * metadata CSV file

---

## HAR Harmonisation

PAMAP2 and WISDM are harmonised to:

* Sampling rate: **20 Hz**
* Channel schema:

  ```
  accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
  ```
* Window definitions:

  * Pretraining: 10 seconds (no overlap)
  * Supervised: 5 seconds (50% overlap)

Note:
WISDM “stairs” activity does not distinguish direction and was mapped to a unified stair-compatible class.

---

## EEG Processing

* Dataset: EEGMMIDB
* Runs used: **R04, R08, R12**
* Sampling rate: **160 Hz (native)**
* Windows: 4-second segments aligned to **T1 and T2 events**

---

## ECG Processing

* Dataset: PTB-XL
* Sampling rate: **100 Hz**
* Signals: full 10-second ECG records
* Splits:

  * `test` (fold 10)
  * `train_cv_1` to `train_cv_9`

---

## Validation

Validation includes:

* metadata completeness checks
* array integrity (no NaN/inf)
* HAR harmonisation verification
* EEG event validation
* ECG fold validation

---

## Resource Estimate

See:

```
reports/resource_estimate.txt
```

Summary:

* Raw data: ~14.3 GB
* Processed data: ~11.6 MB
* Peak RAM: < 2 GB
* Runtime: ~10 minutes

---

## Reproducibility

The pipeline can be executed end-to-end from a clean directory using:

```
python setup_data.py
python preprocess.py --dataset pamap2
python preprocess.py --dataset wisdm
python preprocess.py --dataset eegmmidb
python preprocess.py --dataset ptbxl
python validate_outputs.py
python smoke_tests.py
```

---

## Notes

* Only representative sample packs are included for submission.
* Full processed datasets are not required and were intentionally excluded.
* The pipeline is designed to be scalable and memory-efficient.
