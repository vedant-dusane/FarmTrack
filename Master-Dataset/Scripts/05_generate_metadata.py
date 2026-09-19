# ---------------------------------------------------------------------------
# Example run:
# python 05_generate_metadata.py
#
# Requires: pip install pillow
# ---------------------------------------------------------------------------

"""
05_generate_metadata.py

Purpose (per roadmap step 6 — Metadata Generation)
------------------------------------------------------
Performs a genuine fresh data study of the final 15-breed dataset: scans
each breed folder directly on disk (disk is treated as ground truth),
opens every image file, and records real width/height/format/color-mode/
quality info — rather than trusting or recompiling any previous CSV.

The old final_15breed_manifest.csv (from 04b) is used only as a REFERENCE
for provenance (role: unique/canonical/augmented/augmented_topup) — not as
an authority on what exists. Specifically:

    - Files found on disk but NOT in the old manifest -> kept, added,
      flagged role='discovered_unlisted' (e.g. from manual folder cleanup
      or restoration between steps).
    - Files in the old manifest but NOT found on disk anymore -> dropped
      from the new manifest, logged in reconciliation_report.csv.

Output
------
- master_metadata.csv: in-depth per-image reference table —
    path, breed, dataset_code, source_type, role, width, height, format,
    mode, is_grayscale, aspect_ratio, resolution_category,
    file_size_bytes, status
- final_15breed_manifest.csv: UPDATED to match disk reality (path, breed,
  role). The old version is backed up first (never overwritten blind).
- reconciliation_report.csv: exactly what changed vs. the old manifest
  (added / removed / unchanged counts, per breed).

Usage
-----
1. Edit MASTER_DATASET_ROOT and SELECTED_BREEDS below.
2. Run: python 05_generate_metadata.py
3. Review reconciliation_report.csv and the console summary.
"""

import csv
import shutil
from pathlib import Path
from collections import defaultdict

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    raise SystemExit("Missing dependency. Install with:\n    pip install pillow")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

MASTER_DATASET_ROOT = r"D:\Projects\Federated_Learning\Master-Dataset\master_dataset"

OLD_MANIFEST_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\final_15breed_manifest.csv"
MANIFEST_BACKUP_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\final_15breed_manifest_backup.csv"

MASTER_METADATA_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\master_metadata.csv"
RECONCILIATION_REPORT_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\reconciliation_report.csv"

SELECTED_BREEDS = [
    "holstein-friesian", "gir", "sahiwal", "jersey", "ayrshire",
    "brown-swiss", "jaffarabadi", "nagori", "nili-ravi", "tharparkar",
    "kankrej", "umblachery", "rathi", "red-sindhi", "vechur",
]

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# resolution category thresholds, based on the SHORTER side of the image
SMALL_MAX = 200
MEDIUM_MAX = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def infer_dataset_code(filename: str):
    """Filenames from 02_merge_datasets.py are prefixed 'DSxx__original_name'."""
    if "__" in filename:
        prefix = filename.split("__", 1)[0]
        if prefix.upper().startswith("DS") and prefix[2:].isdigit():
            return prefix.upper()
    return "unknown"


def infer_source_type(filename: str, role: str):
    lower = filename.lower()
    if "__topup" in lower or role == "augmented_topup":
        return "synthetic_topup"
    if "__aug" in lower or role == "augmented":
        return "synthetic_augmented"
    if role in ("unique", "canonical"):
        return "original"
    return "unknown"


def resolution_category(width, height):
    shorter = min(width, height)
    if shorter < SMALL_MAX:
        return "small"
    elif shorter < MEDIUM_MAX:
        return "medium"
    return "large"


def inspect_image(path: Path):
    """Actually opens the file. Returns dict of real attributes or status='corrupt'."""
    try:
        with Image.open(path) as img:
            img.load()
            width, height = img.size
            img_format = img.format
            mode = img.mode
        file_size = path.stat().st_size
        return {
            "status": "ok",
            "width": width, "height": height,
            "format": img_format, "mode": mode,
            "file_size_bytes": file_size,
        }
    except (UnidentifiedImageError, OSError, SyntaxError, Exception) as e:
        return {"status": "corrupt", "detail": str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load old manifest as a reference lookup (path -> role)
    old_manifest = {}
    try:
        with open(OLD_MANIFEST_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                old_manifest[row["path"]] = row["role"]
        shutil.copy(OLD_MANIFEST_CSV, MANIFEST_BACKUP_CSV)
        print(f"Backed up old manifest to: {MANIFEST_BACKUP_CSV}\n")
    except FileNotFoundError:
        print("No old manifest found — proceeding with disk scan only.\n")

    metadata_rows = []
    new_manifest_rows = []
    added_paths = []
    corrupt_paths = []

    print("Scanning breed folders on disk (this actually opens every file)...\n")

    for breed in SELECTED_BREEDS:
        breed_dir = Path(MASTER_DATASET_ROOT) / breed
        if not breed_dir.exists():
            print(f"  [WARNING] breed folder not found: {breed_dir}")
            continue

        files = [p for p in breed_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS]
        print(f"Scanning: {breed} ({len(files)} files on disk)")

        for path in files:
            path_str = str(path)
            role = old_manifest.get(path_str)
            if role is None:
                role = "discovered_unlisted"
                added_paths.append(path_str)

            info = inspect_image(path)
            if info["status"] == "corrupt":
                corrupt_paths.append(path_str)
                metadata_rows.append({
                    "path": path_str, "breed": breed,
                    "dataset_code": infer_dataset_code(path.name),
                    "source_type": infer_source_type(path.name, role),
                    "role": role, "width": "", "height": "", "format": "",
                    "mode": "", "is_grayscale": "", "aspect_ratio": "",
                    "resolution_category": "", "file_size_bytes": "",
                    "status": "corrupt",
                })
                continue

            width, height, mode = info["width"], info["height"], info["mode"]
            metadata_rows.append({
                "path": path_str, "breed": breed,
                "dataset_code": infer_dataset_code(path.name),
                "source_type": infer_source_type(path.name, role),
                "role": role,
                "width": width, "height": height,
                "format": info["format"], "mode": mode,
                "is_grayscale": mode in ("L", "LA"),
                "aspect_ratio": round(width / height, 3) if height else "",
                "resolution_category": resolution_category(width, height),
                "file_size_bytes": info["file_size_bytes"],
                "status": "ok",
            })
            new_manifest_rows.append({"path": path_str, "breed": breed, "role": role})

    # Reconciliation: what was in old manifest but not found on disk
    disk_paths = {row["path"] for row in new_manifest_rows} | set(corrupt_paths)
    removed_paths = [p for p in old_manifest if p not in disk_paths]

    # --- Write outputs ---
    with open(MASTER_METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "path", "breed", "dataset_code", "source_type", "role",
            "width", "height", "format", "mode", "is_grayscale",
            "aspect_ratio", "resolution_category", "file_size_bytes", "status"
        ])
        writer.writeheader()
        writer.writerows(metadata_rows)

    with open(OLD_MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "breed", "role"])
        writer.writeheader()
        writer.writerows(new_manifest_rows)

    recon_rows = []
    for p in added_paths:
        recon_rows.append({"path": p, "change": "added_discovered_on_disk"})
    for p in removed_paths:
        recon_rows.append({"path": p, "change": "removed_missing_from_disk"})
    for p in corrupt_paths:
        recon_rows.append({"path": p, "change": "corrupt_kept_in_metadata_excluded_from_manifest"})

    with open(RECONCILIATION_REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "change"])
        writer.writeheader()
        writer.writerows(recon_rows)

    # --- Summary ---
    breed_counts = defaultdict(int)
    grayscale_count = 0
    res_counts = defaultdict(int)
    for row in metadata_rows:
        if row["status"] == "ok":
            breed_counts[row["breed"]] += 1
            if row["is_grayscale"]:
                grayscale_count += 1
            res_counts[row["resolution_category"]] += 1

    print("\n" + "=" * 60)
    print("METADATA GENERATION SUMMARY")
    print("=" * 60)
    total = 0
    for breed in SELECTED_BREEDS:
        c = breed_counts.get(breed, 0)
        total += c
        print(f"{breed:22s} {c}")
    print("-" * 60)
    print(f"{'TOTAL (ok)':22s} {total}")
    print(f"\nCorrupt files found: {len(corrupt_paths)}")
    print(f"Grayscale images: {grayscale_count}")
    print(f"Resolution breakdown: {dict(res_counts)}")
    print(f"\nReconciliation vs old manifest:")
    print(f"  Added (found on disk, not in old manifest): {len(added_paths)}")
    print(f"  Removed (in old manifest, missing from disk): {len(removed_paths)}")
    print(f"\nmaster_metadata.csv:         {MASTER_METADATA_CSV}")
    print(f"Updated manifest:             {OLD_MANIFEST_CSV}")
    print(f"Old manifest backup:          {MANIFEST_BACKUP_CSV}")
    print(f"Reconciliation report:        {RECONCILIATION_REPORT_CSV}")


if __name__ == "__main__":
    main()
