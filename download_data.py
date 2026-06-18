"""
============================================================
download_data.py
Downloads the Chest X-Ray Pneumonia dataset from Kaggle
============================================================

Dataset:
  Name   : Chest X-Ray Images (Pneumonia)
  Source : https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
  Size   : ~2 GB
  Classes: NORMAL, PNEUMONIA
  Split  : train / val / test (pre-split by Kaggle)

Prerequisites:
  1. Install kaggle CLI: pip install kaggle
  2. Get API token from: https://www.kaggle.com/settings → API → Create New Token
  3. Place kaggle.json in ~/.kaggle/kaggle.json (chmod 600)

Usage:
    python download_data.py
"""

import os
import subprocess
import zipfile
from pathlib import Path

RAW_DIR    = Path("data/raw")
DATASET_ID = "paultimothymooney/chest-xray-pneumonia"
ZIP_NAME   = "chest-xray-pneumonia.zip"


def download_kaggle_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / ZIP_NAME

    if not zip_path.exists():
        print(f"Downloading {DATASET_ID} via Kaggle API …")
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", DATASET_ID,
                "-p", str(RAW_DIR),
            ],
            check=True,
        )
        print("Download complete.")
    else:
        print(f"ZIP already exists at {zip_path}, skipping download.")

    # Unzip
    extract_dir = RAW_DIR / "chest_xray"
    if not extract_dir.exists():
        print(f"Extracting {zip_path} …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RAW_DIR)
        print(f"Extracted to {extract_dir}")

        # The Kaggle zip extracts as 'chest_xray' which already has train/val/test
        # Fix: Kaggle sometimes nests it one level deeper
        nested = RAW_DIR / "chest_xray" / "chest_xray"
        if nested.exists():
            import shutil
            for item in nested.iterdir():
                shutil.move(str(item), str(extract_dir))
            nested.rmdir()
    else:
        print(f"Data already extracted at {extract_dir}.")

    # Verify
    for split in ["train", "val", "test"]:
        split_dir = extract_dir / split
        if split_dir.exists():
            n_files = sum(1 for _ in split_dir.rglob("*.jpeg"))
            print(f"  {split}: {n_files} images")
        else:
            print(f"  WARNING: {split_dir} not found!")


if __name__ == "__main__":
    download_kaggle_dataset()
