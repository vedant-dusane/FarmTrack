# ---------------------------------------------------------------------------
# Example run:
# python 02_merge_datasets.py
# (edit DATASET_ROOTS below first, same paths you used in 01_scan_datasets.py)
# ---------------------------------------------------------------------------

"""

Purpose
-------
Merges all 11 raw cattle/buffalo datasets (breed folders already standardized
to match your Excel taxonomy) into one master, training-ready structure:

    master_dataset/
        Gir/
        Sahiwal/
        Murrah/
        ...

Also produces metadata.csv, which becomes the single source of truth for
every image going forward (breed, originating dataset, original filename,
new filename/path). Later scripts (validation, dedup, split) should read
FROM metadata.csv, not re-scan folders.

Design decisions
----------------
- COPIES files, never moves/deletes originals. Your raw datasets are
  never touched or modified.
- Filenames are prefixed with the source dataset name to guarantee no
  collisions when multiple datasets contribute to the same breed folder,
  e.g.: Gir/BovCap-5K__cow_0231.jpg
- RESUMABLE: if a destination file already exists, it's skipped instead of
  re-copied. Safe to re-run after a crash/interruption on ~48k images.
- Does NOT validate/open images (corrupt file checks happen in
  03_validate_images.py) — this script only copies bytes, keeping it fast
  and simple, one responsibility only.

Usage
-----
1. Edit DATASET_ROOTS below (same dict you used in 01_scan_datasets.py).
2. Edit MASTER_DATASET_ROOT to where you want the merged dataset to live.
3. Run: python 02_merge_datasets.py
4. Check the printed summary against your reconciled inventory numbers.
"""

import csv
import shutil
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONFIG — fill this in (reuse the same paths from 01_scan_datasets.py)
# ---------------------------------------------------------------------------

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

# Short codes used in filenames instead of full dataset names, to avoid
# Windows MAX_PATH (260 char) issues once combined with breed name + master
# dataset folder path. Full dataset name is still preserved in metadata.csv.
DATASET_CODES = {
    "A-Comprehensive-Visual-Dataset-of-17-Indian-Government-Recognized-Buffalo-Breeds": "DS01",
    "A-Comprehensive-Visual-Dataset-of-50-Government-Recognized-Indian-Cattle-Breeds": "DS02",
    "BovCap-5K-Annotated-Bovine-Image-Caption-Dataset": "DS03",
    "Breed-cattle-buffalo": "DS04",
    "Catbuf-dataset-Dataset-of-cows-and-Buffalo": "DS05",
    "CATTLE-BREED-RECOGNITION": "DS06",
    "Cattle-Breeds-Dataset": "DS07",
    "Cattle-Buffalo-breeds-Computer-Vision-Model": "DS08",
    "Cow-Breed-Classification-Dataset": "DS09",
    "Indian-Bovine-breeds": "DS10",
    "Indian-Cattle-&-Buffalo-Breeds-Dataset": "DS11",
}

# Where the merged, standardized dataset should be created
MASTER_DATASET_ROOT = r"D:\Projects\Federated_Learning\Master-Dataset\master_dataset"

# Where metadata.csv should be written
METADATA_CSV_PATH = r"D:\Projects\Federated_Learning\Master-Dataset\metadata.csv"

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------------------------

def merge_breed_folder(dataset_name, dataset_code, breed_folder: Path, dest_breed_dir: Path, metadata_rows):
    """
    Copies every valid image from one raw breed folder into the master
    dataset's corresponding breed folder, prefixing the filename with the
    SHORT dataset code (not the full dataset name) to avoid collisions
    while keeping paths short enough for Windows MAX_PATH limits. The full
    dataset name is preserved in metadata.csv for traceability.
    Returns the number of images copied (new) and skipped (already existed).
    """
    dest_breed_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for entry in sorted(breed_folder.iterdir()):
        if not entry.is_file():
            continue
        ext = entry.suffix.lower()
        if ext not in VALID_IMAGE_EXTENSIONS:
            continue

        new_filename = f"{dataset_code}__{entry.name}"
        dest_path = dest_breed_dir / new_filename

        if dest_path.exists():
            skipped += 1
        else:
            shutil.copy2(entry, dest_path)
            copied += 1

        metadata_rows.append({
            "breed": dest_breed_dir.name,
            "dataset_code": dataset_code,
            "dataset_full_name": dataset_name,
            "original_filename": entry.name,
            "master_filename": new_filename,
            "master_path": str(dest_path),
        })

    return copied, skipped


def main():
    master_root = Path(MASTER_DATASET_ROOT)
    master_root.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    dataset_copied_totals = defaultdict(int)
    dataset_skipped_totals = defaultdict(int)
    breed_totals = defaultdict(int)

    print("Merging datasets into master_dataset/ ...\n")

    for dataset_name, dataset_root in DATASET_ROOTS.items():
        dataset_code = DATASET_CODES.get(dataset_name)
        if not dataset_code:
            print(f"[WARNING] No DS code defined for '{dataset_name}', skipping.")
            continue

        root = Path(dataset_root)
        if not root.exists():
            print(f"[WARNING] Path does not exist, skipping: {root}")
            continue

        breed_folders = [p for p in root.iterdir() if p.is_dir()]
        print(f"Dataset: {dataset_name} ({dataset_code}) — {len(breed_folders)} breed folders")

        for breed_folder in sorted(breed_folders):
            breed_name = breed_folder.name
            dest_breed_dir = master_root / breed_name

            copied, skipped = merge_breed_folder(
                dataset_name, dataset_code, breed_folder, dest_breed_dir, metadata_rows
            )

            dataset_copied_totals[dataset_name] += copied
            dataset_skipped_totals[dataset_name] += skipped
            breed_totals[breed_name] += copied + skipped

        print(f"  -> copied {dataset_copied_totals[dataset_name]}, "
              f"skipped (already existed) {dataset_skipped_totals[dataset_name]}\n")

    # Write metadata.csv
    metadata_path = Path(METADATA_CSV_PATH)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["breed", "dataset_code", "dataset_full_name",
                        "original_filename", "master_filename", "master_path"]
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    # Summary
    print("=" * 70)
    print("MERGE SUMMARY (cross-check against your reconciled inventory)")
    print("=" * 70)
    grand_copied = 0
    grand_skipped = 0
    for dataset_name in DATASET_ROOTS:
        c = dataset_copied_totals.get(dataset_name, 0)
        s = dataset_skipped_totals.get(dataset_name, 0)
        grand_copied += c
        grand_skipped += s
        print(f"{dataset_name:70s} copied={c:6d}  skipped={s:6d}")
    print("-" * 70)
    print(f"{'TOTAL':70s} copied={grand_copied:6d}  skipped={grand_skipped:6d}")
    print(f"{'GRAND TOTAL IN MASTER DATASET':70s} {grand_copied + grand_skipped}")
    print(f"\nBreed folders created: {len(breed_totals)}")
    print(f"metadata.csv written to: {metadata_path}")
    print(f"master_dataset/ created at: {master_root}")


if __name__ == "__main__":
    main()
