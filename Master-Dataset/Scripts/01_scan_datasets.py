"""
01_scan_datasets.py

Purpose
-------
Scans all raw cattle/buffalo image datasets (where breed folders have already
been manually renamed to match the standardized breed names used in the
master Excel inventory) and produces a machine-readable inventory CSV.

This CSV is meant to be cross-checked against the manually built Excel
inventory (Sheet 1: "Images Per Set") to catch mismatches caused by manual
errors — miscounts, stray files, case-sensitivity issues, misplaced images,
non-image files sitting inside breed folders, etc.

Output
------
dataset_inventory.csv with columns:
    dataset, breed, image_count, formats, corrupt_or_zero_byte_count

Usage
-----
1. Edit the DATASET_ROOTS dictionary below with your 11 dataset paths.
2. Run: python 01_scan_datasets.py
3. Check dataset_inventory.csv (and the console summary) against your Excel.
"""

import os
import csv
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONFIG — fill this in with your actual paths
# ---------------------------------------------------------------------------
# Key   = a short name for the dataset (should match the dataset name used
#         as a column header in your Excel "Images Per Set" sheet)
# Value = path to the ROOT folder of that dataset, i.e. the folder that
#         directly contains the breed subfolders (Gir/, Sahiwal/, Murrah/, ...)

DATASET_ROOTS = {
    "A-Comprehensive-Visual-Dataset-of-17-Indian-Government-Recognized-Buffalo-Breeds": r"Unzip-Modified\A-Comprehensive-Visual-Dataset-of-17-Indian-Government-Recognized-Buffalo-Breeds\buffalo",
    "A-Comprehensive-Visual-Dataset-of-50-Government-Recognized-Indian-Cattle-Breeds": r"Unzip-Modified\A-Comprehensive-Visual-Dataset-of-50-Government-Recognized-Indian-Cattle-Breeds\cattle",
    "BovCap-5K-Annotated-Bovine-Image-Caption-Dataset": r"Unzip-Modified\BovCap-5K-Annotated-Bovine-Image-Caption-Dataset\Images",
    "Breed-cattle-buffalo": r"Unzip-Modified\Breed-cattle-buffalo\clean_images\buffalo",
    "Catbuf-dataset-Dataset-of-cows-and-Buffalo": r"Unzip-Modified\Catbuf-dataset-Dataset-of-cows-and-Buffalo\yolov8",
    "CATTLE-BREED-RECOGNITION": r"Unzip-Modified\CATTLE-BREED-RECOGNITION\dataset\train",
    "Cattle-Breeds-Dataset": r"Unzip-Modified\Cattle-Breeds-Dataset\Cattle Breeds",
    "Cattle-Buffalo-breeds-Computer-Vision-Model": r"Unzip-Modified\Cattle-Buffalo-breeds-Computer-Vision-Model\train",
    "Cow-Breed-Classification-Dataset": r"Unzip-Modified\Cow-Breed-Classification-Dataset\Cow Breed Dataset",
    "Indian-Bovine-breeds": r"Unzip-Modified\Indian-Bovine-breeds\Indian_bovine_breeds\Indian_bovine_breeds",
    "Indian-Cattle-&-Buffalo-Breeds-Dataset": r"Unzip-Modified\Indian-Cattle-&-Buffalo-Breeds-Dataset\IndianCattleBuffaloeBreeds-Dataset",
}

# Recognized image extensions (lowercase). Add more if your datasets use them.
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

OUTPUT_CSV = "D:\Projects\Federated_Learning\Master-Dataset\Scripts\Script_CountOf_Dataset.csv"


# ---------------------------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------------------------

def scan_breed_folder(folder_path: Path):
    """
    Scans one breed folder and returns:
        image_count, set_of_formats_found, zero_byte_or_suspicious_count
    Does NOT open/decode images (that's 03_validate_images.py's job) —
    this script only counts and flags obvious problems (0-byte files),
    keeping this step fast for a first-pass inventory.
    """
    image_count = 0
    formats_found = set()
    zero_byte_count = 0

    for entry in folder_path.iterdir():
        if not entry.is_file():
            continue
        ext = entry.suffix.lower()
        if ext not in VALID_IMAGE_EXTENSIONS:
            continue
        image_count += 1
        formats_found.add(ext)
        try:
            if entry.stat().st_size == 0:
                zero_byte_count += 1
        except OSError:
            zero_byte_count += 1

    return image_count, formats_found, zero_byte_count


def scan_dataset(dataset_name: str, dataset_root: str):
    """
    Scans one dataset root folder. Each subfolder inside is treated as a
    breed folder (folder name == standardized breed name, per your manual
    renaming). Returns a list of row dicts, one per breed.
    """
    root = Path(dataset_root)
    rows = []

    if not root.exists():
        print(f"  [WARNING] Path does not exist, skipping: {root}")
        return rows

    breed_folders = [p for p in root.iterdir() if p.is_dir()]

    if not breed_folders:
        print(f"  [WARNING] No breed subfolders found in: {root}")
        return rows

    for breed_folder in sorted(breed_folders):
        breed_name = breed_folder.name
        image_count, formats_found, zero_byte_count = scan_breed_folder(breed_folder)

        rows.append({
            "dataset": dataset_name,
            "breed": breed_name,
            "image_count": image_count,
            "formats": "|".join(sorted(formats_found)) if formats_found else "",
            "zero_byte_or_suspicious_count": zero_byte_count,
        })

    return rows


def main():
    all_rows = []
    dataset_totals = defaultdict(int)
    dataset_breed_counts = defaultdict(int)

    print("Scanning datasets...\n")

    for dataset_name, dataset_root in DATASET_ROOTS.items():
        print(f"Scanning: {dataset_name}")
        rows = scan_dataset(dataset_name, dataset_root)
        all_rows.extend(rows)

        for row in rows:
            dataset_totals[dataset_name] += row["image_count"]
            if row["image_count"] > 0:
                dataset_breed_counts[dataset_name] += 1

        print(f"  -> {len(rows)} breed folders found, "
              f"{dataset_totals[dataset_name]} images total\n")

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "breed", "image_count", "formats",
                        "zero_byte_or_suspicious_count"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    # Console summary (compare these numbers against your Excel "Data Count" sheet)
    print("=" * 60)
    print("SUMMARY (cross-check against your Excel 'Data Count' sheet)")
    print("=" * 60)
    grand_total = 0
    for dataset_name in DATASET_ROOTS:
        total = dataset_totals.get(dataset_name, 0)
        breeds = dataset_breed_counts.get(dataset_name, 0)
        grand_total += total
        print(f"{dataset_name:70s} images={total:6d}  breeds={breeds:3d}")
    print("-" * 60)
    print(f"{'TOTAL IMAGES ACROSS ALL DATASETS':70s} {grand_total}")
    print(f"\nInventory written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
