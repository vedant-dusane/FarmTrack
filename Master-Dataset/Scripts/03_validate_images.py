# ---------------------------------------------------------------------------
# Example run:
# python 03_validate_images.py
# (edit METADATA_CSV_PATH below if it's not in the default location)
#
# Requires: pip install pillow
# ---------------------------------------------------------------------------

"""
03_validate_images.py

Purpose (per roadmap step 3 — Image Validation)
-------------------------------------------------
Checks every image in master_dataset/ (via metadata.csv from
02_merge_datasets.py) for:
    - Corrupt files (fails to load at all)
    - Wrong formats (actual image format doesn't match file extension)
    - Resolution issues (too small to be a real usable photo)
    - Readability (opens but fails a full-load consistency check)

Does NOT modify, move, or delete any files — purely reports. Produces a
SEPARATE validation_report.csv (kept independent from metadata.csv, since
this will be re-run and overwritten independently as issues get fixed).

Output
------
validation_report.csv, one row per image:
    master_path, breed, status, width, height, format, mode, detail

status is one of: ok, corrupt, wrong_format, too_small, unreadable

Usage
-----
1. Edit METADATA_CSV_PATH to point at the metadata.csv from 02_merge_datasets.py.
2. Adjust MIN_WIDTH / MIN_HEIGHT if your definition of "too small" differs.
3. Run: python 03_validate_images.py
4. Review validation_report.csv — filter for status != 'ok' to see problem files.
"""

import csv
from pathlib import Path

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    raise SystemExit("Missing dependency. Install with:\n    pip install pillow")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

METADATA_CSV_PATH = r"D:\Projects\Federated_Learning\Master-Dataset\metadata.csv"
VALIDATION_REPORT_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\validation_report.csv"

# Images smaller than this on either dimension are flagged as "too_small"
# (likely thumbnails/icons/broken crops, not usable training photos)
MIN_WIDTH = 50
MIN_HEIGHT = 50

# Maps file extension -> expected Pillow format string(s)
EXTENSION_TO_FORMAT = {
    ".jpg": {"JPEG"},
    ".jpeg": {"JPEG"},
    ".png": {"PNG"},
    ".bmp": {"BMP"},
    ".tif": {"TIFF"},
    ".tiff": {"TIFF"},
    ".webp": {"WEBP"},
}


# ---------------------------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------------------------

def validate_image(path: Path):
    """
    Returns a dict: status, width, height, format, mode, detail
    Never raises — always returns a result, even for totally broken files.
    """
    result = {
        "status": "ok",
        "width": None,
        "height": None,
        "format": None,
        "mode": None,
        "detail": "",
    }

    # Step 1: try to open + verify (verify() checks structural integrity,
    # but the file must be re-opened afterward for further use)
    try:
        with Image.open(path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        result["status"] = "corrupt"
        result["detail"] = str(e)
        return result

    # Step 2: re-open (verify() invalidates the file object) and do a full
    # load to catch truncated-data issues that verify() alone can miss
    try:
        with Image.open(path) as img:
            img.load()
            width, height = img.size
            img_format = img.format
            mode = img.mode
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        result["status"] = "unreadable"
        result["detail"] = str(e)
        return result

    result["width"] = width
    result["height"] = height
    result["format"] = img_format
    result["mode"] = mode

    # Step 3: format-vs-extension mismatch check
    ext = path.suffix.lower()
    expected_formats = EXTENSION_TO_FORMAT.get(ext)
    if expected_formats and img_format not in expected_formats:
        result["status"] = "wrong_format"
        result["detail"] = f"extension={ext} but actual format={img_format}"
        return result

    # Step 4: resolution check
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        result["status"] = "too_small"
        result["detail"] = f"{width}x{height} below minimum {MIN_WIDTH}x{MIN_HEIGHT}"
        return result

    return result


def main():
    rows = []
    with open(METADATA_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Validating {len(rows)} images...\n")

    report_rows = []
    status_counts = {"ok": 0, "corrupt": 0, "wrong_format": 0, "too_small": 0, "unreadable": 0}

    for i, row in enumerate(rows, start=1):
        path = Path(row["master_path"])
        result = validate_image(path)
        status_counts[result["status"]] += 1

        report_rows.append({
            "master_path": str(path),
            "breed": row["breed"],
            "status": result["status"],
            "width": result["width"],
            "height": result["height"],
            "format": result["format"],
            "mode": result["mode"],
            "detail": result["detail"],
        })

        if i % 2000 == 0:
            print(f"  ...processed {i}/{len(rows)}")

    # Write validation_report.csv
    with open(VALIDATION_REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["master_path", "breed", "status", "width", "height",
                        "format", "mode", "detail"]
        )
        writer.writeheader()
        writer.writerows(report_rows)

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for status, count in status_counts.items():
        print(f"{status:15s} {count}")
    print("-" * 60)
    print(f"{'TOTAL':15s} {len(rows)}")
    print(f"\nFull report written to: {VALIDATION_REPORT_CSV}")
    print("Filter for status != 'ok' to see all problem files before step 4 (dedup).")


if __name__ == "__main__":
    main()
