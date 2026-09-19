#!/usr/bin/env python3
"""
Automated downloader for the Cattle Breed Classification Dataset from Kaggle.
Dataset URL: https://www.kaggle.com/datasets/vedant0dusane/cattle-breed-classification-dataset
"""

import os
import sys
import shutil
import zipfile
import argparse
from pathlib import Path

DATASET_SLUG = "vedant0dusane/cattle-breed-classification-dataset"

def download_with_kagglehub(target_dir: Path):
    print(f"[*] Downloading dataset via kagglehub: {DATASET_SLUG}...")
    try:
        import kagglehub
        downloaded_path = kagglehub.dataset_download(DATASET_SLUG)
        print(f"[+] Downloaded successfully to cache: {downloaded_path}")
        
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"[*] Copying files to destination: {target_dir}...")
        for item in Path(downloaded_path).iterdir():
            dest = target_dir / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        print(f"[+] Dataset is ready at: {target_dir.resolve()}")
    except ImportError:
        print("[-] 'kagglehub' is not installed. Run: pip install kagglehub")
        return False
    except Exception as e:
        print(f"[-] kagglehub download failed: {e}")
        return False
    return True

def download_with_kaggle_api(target_dir: Path):
    print(f"[*] Downloading dataset via Kaggle API: {DATASET_SLUG}...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        target_dir.mkdir(parents=True, exist_ok=True)
        zip_path = target_dir / "cattle_dataset.zip"
        
        print(f"[*] Downloading archive to {zip_path}...")
        api.dataset_download_files(DATASET_SLUG, path=str(target_dir), unzip=True, quiet=False)
        print(f"[+] Downloaded and extracted successfully to: {target_dir.resolve()}")
    except ImportError:
        print("[-] 'kaggle' CLI/library not found. Run: pip install kaggle")
        return False
    except Exception as e:
        print(f"[-] Kaggle API download failed: {e}")
        print("[!] Note: Make sure ~/.kaggle/kaggle.json (or C:\\Users\\<user>\\.kaggle\\kaggle.json) exists with your API key.")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Download Cattle Breed Classification Dataset from Kaggle")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Master-Dataset/master_dataset",
        help="Destination directory to unpack the dataset (default: Master-Dataset/master_dataset)"
    )
    parser.add_argument(
        "--method",
        choices=["kagglehub", "kaggle-api", "auto"],
        default="auto",
        help="Download method (default: auto)"
    )
    args = parser.parse_args()
    target_dir = Path(args.output_dir)

    success = False
    if args.method in ("kagglehub", "auto"):
        success = download_with_kagglehub(target_dir)
    
    if not success and args.method in ("kaggle-api", "auto"):
        success = download_with_kaggle_api(target_dir)

    if not success:
        print("\n" + "="*70)
        print("[!] Automated download could not be completed.")
        print("You can manually download the dataset using either:")
        print(f"  1. Kaggle CLI: kaggle datasets download -d {DATASET_SLUG} --unzip -p {target_dir}")
        print(f"  2. Browser: Visit https://www.kaggle.com/datasets/{DATASET_SLUG}")
        print(f"     and extract the downloaded files into: {target_dir.resolve()}")
        print("="*70)
        sys.exit(1)

if __name__ == "__main__":
    main()
