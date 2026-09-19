# ---------------------------------------------------------------------------
# Example run:
# python 06a_verify_grayscale.py
#
# Requires: pip install pillow numpy
# ---------------------------------------------------------------------------

"""
06a_verify_grayscale.py

Purpose
-------
Corrects a real limitation found in 06_dataset_statistics.py's grayscale
check: that check only looked at PIL storage mode (mode == 'L'/'LA'), but
many visually black-and-white/grayscale photos are still SAVED as regular
RGB files (R≈G≈B per pixel) — very common from scrapers/tools that default
to RGB on save. This meant DS08 (the original "Computer Vision" grayscale
dataset) was contributing real images while showing up as 0% grayscale.

This script re-opens every image and checks ACTUAL pixel content: if the
color channels are nearly identical across the image (mean channel
difference below GRAYSCALE_DIFF_THRESHOLD), it's classified as visually
grayscale regardless of file mode.

Output
------
- master_metadata.csv: is_grayscale column CORRECTED in place (old version
  backed up first as master_metadata_backup.csv). A new
  grayscale_detection_method column is added for transparency (mode vs
  channel_analysis vs not_grayscale).
- grayscale_correction_report.csv: only the images where this check
  DISAGREES with the old mode-based flag — i.e. the false negatives just
  found. Full audit trail, not a silent overwrite.

After running this, re-run 06_dataset_statistics.py to regenerate accurate
grayscale stats/charts from the corrected master_metadata.csv.

Usage
-----
1. Edit MASTER_METADATA_CSV below.
2. Adjust GRAYSCALE_DIFF_THRESHOLD if needed (0-255 scale; lower = stricter).
3. Run: python 06a_verify_grayscale.py
"""

import csv
import shutil
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
except ImportError:
    raise SystemExit("Missing dependencies. Install with:\n    pip install pillow numpy")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

MASTER_METADATA_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\master_metadata.csv"
METADATA_BACKUP_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\master_metadata_backup.csv"
CORRECTION_REPORT_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\grayscale_correction_report.csv"

GRAYSCALE_DIFF_THRESHOLD = 8   # mean pairwise channel diff (0-255 scale) below this = grayscale
DOWNSAMPLE_SIZE = (64, 64)      # resize before analysis, for speed


# ---------------------------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------------------------

def check_visual_grayscale(path: Path):
    """
    Returns (is_grayscale: bool, method: str, mean_channel_diff: float or None)
    method is 'mode' (already single-channel), 'channel_analysis' (RGB but
    visually grayscale), or 'not_grayscale'.
    Returns (None, 'unreadable', None) on failure.
    """
    try:
        with Image.open(path) as img:
            if img.mode in ("L", "LA", "1"):
                return True, "mode", 0.0

            rgb = img.convert("RGB").resize(DOWNSAMPLE_SIZE)
            arr = np.asarray(rgb, dtype=np.int16)
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            diff = (np.abs(r - g).mean() + np.abs(g - b).mean() + np.abs(r - b).mean()) / 3
            is_gray = diff < GRAYSCALE_DIFF_THRESHOLD
            return is_gray, ("channel_analysis" if is_gray else "not_grayscale"), float(diff)
    except Exception:
        return None, "unreadable", None


def main():
    with open(MASTER_METADATA_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())
    if "grayscale_detection_method" not in fieldnames:
        fieldnames.append("grayscale_detection_method")

    shutil.copy(MASTER_METADATA_CSV, METADATA_BACKUP_CSV)
    print(f"Backed up old metadata to: {METADATA_BACKUP_CSV}\n")
    print(f"Re-checking {len(rows)} images for actual visual grayscale content...\n")

    correction_rows = []
    old_gray_count = 0
    new_gray_count = 0
    unreadable_count = 0

    for i, row in enumerate(rows, start=1):
        if row["status"] != "ok":
            row["grayscale_detection_method"] = "skipped_not_ok"
            continue

        old_flag = row["is_grayscale"] == "True"
        if old_flag:
            old_gray_count += 1

        path = Path(row["path"])
        is_gray, method, diff = check_visual_grayscale(path)

        if is_gray is None:
            unreadable_count += 1
            row["grayscale_detection_method"] = "unreadable_at_recheck"
            continue

        row["is_grayscale"] = str(is_gray)
        row["grayscale_detection_method"] = method
        if is_gray:
            new_gray_count += 1

        if is_gray != old_flag:
            correction_rows.append({
                "path": str(path), "breed": row["breed"],
                "dataset_code": row["dataset_code"],
                "old_is_grayscale": old_flag, "new_is_grayscale": is_gray,
                "method": method, "mean_channel_diff": round(diff, 2) if diff is not None else "",
            })

        if i % 2000 == 0:
            print(f"  ...processed {i}/{len(rows)}")

    with open(MASTER_METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(CORRECTION_REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "path", "breed", "dataset_code", "old_is_grayscale",
            "new_is_grayscale", "method", "mean_channel_diff"
        ])
        writer.writeheader()
        writer.writerows(correction_rows)

    print("\n" + "=" * 60)
    print("GRAYSCALE CORRECTION SUMMARY")
    print("=" * 60)
    print(f"Old (mode-based) grayscale count: {old_gray_count}")
    print(f"New (verified) grayscale count:   {new_gray_count}")
    print(f"Corrections made (disagreements): {len(correction_rows)}")
    print(f"Unreadable at re-check:           {unreadable_count}")
    print(f"\nUpdated master_metadata.csv:      {MASTER_METADATA_CSV}")
    print(f"Old metadata backup:              {METADATA_BACKUP_CSV}")
    print(f"Correction report:                {CORRECTION_REPORT_CSV}")
    print("\nRe-run 06_dataset_statistics.py now to regenerate accurate grayscale stats/charts.")


if __name__ == "__main__":
    main()
