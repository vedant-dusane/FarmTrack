# ---------------------------------------------------------------------------
# Example run:
# python 05a_quarantine_unexplained_discovered.py
# ---------------------------------------------------------------------------

"""
05a_quarantine_unexplained_discovered.py

Purpose
-------
Directly targets and quarantines the specific files flagged as
'added_discovered_on_disk' in reconciliation_report.csv (from
05_generate_metadata.py) — the ~7,195 unexplained synthetic_augmented
files that appeared in supposedly-zero-duplicate breeds, likely from an
accidental restore-induced duplication. Does NOT touch any other files.

Steps
-----
1. Read reconciliation_report.csv, take only rows with
   change == 'added_discovered_on_disk'.
2. Look up each flagged path's breed in master_metadata.csv.
3. Move each flagged file to:
     master_dataset/scrapped_photos/unexplained_discovered/<breed>/<filename>
   (moved, never deleted — fully reversible).
4. Rewrite master_metadata.csv EXCLUDING the quarantined rows.
5. Rebuild final_15breed_manifest.csv from the remaining master_metadata.csv
   rows (role != 'discovered_unlisted', status == 'ok') — this should end
   up matching final_15breed_manifest_backup.csv (14,443 rows) if nothing
   else changed.

Usage
-----
1. Edit the CONFIG paths below.
2. Run: python 05a_quarantine_unexplained_discovered.py
3. Confirm the printed final count matches your known-good 14,443.
"""

import csv
import shutil
from pathlib import Path
from collections import defaultdict

MASTER_DATASET_ROOT = r"D:\Projects\Federated_Learning\Master-Dataset\master_dataset"
RECONCILIATION_REPORT_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\reconciliation_report.csv"
MASTER_METADATA_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\master_metadata.csv"
FINAL_15BREED_MANIFEST_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\final_15breed_manifest.csv"

UNEXPLAINED_QUARANTINE_DIR = r"D:\Projects\Federated_Learning\Master-Dataset\scrapped_photos\unexplained_discovered"
QUARANTINE_LOG_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\unexplained_discovered_quarantine_log.csv"


def main():
    # Step 1: flagged paths
    flagged_paths = set()
    with open(RECONCILIATION_REPORT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["change"] == "added_discovered_on_disk":
                flagged_paths.add(row["path"])

    print(f"Flagged paths to quarantine: {len(flagged_paths)}")

    # Step 2/3: load master_metadata, move flagged files, keep the rest
    kept_rows = []
    quarantine_log_rows = []
    moved_count = 0
    missing_count = 0

    with open(MASTER_METADATA_CSV, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    for row in all_rows:
        if row["path"] in flagged_paths:
            src = Path(row["path"])
            breed = row["breed"]
            dest_dir = Path(UNEXPLAINED_QUARANTINE_DIR) / breed
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            moved = False
            if src.exists():
                try:
                    shutil.move(str(src), str(dest))
                    moved = True
                    moved_count += 1
                except Exception as e:
                    print(f"  [WARNING] failed to move {src}: {e}")
            else:
                missing_count += 1
            quarantine_log_rows.append({
                "breed": breed, "original_path": row["path"],
                "new_path": str(dest) if moved else "", "moved": moved,
            })
        else:
            kept_rows.append(row)

    with open(QUARANTINE_LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["breed", "original_path", "new_path", "moved"])
        writer.writeheader()
        writer.writerows(quarantine_log_rows)

    # Step 4: rewrite master_metadata.csv without the quarantined rows
    if kept_rows:
        fieldnames = list(kept_rows[0].keys())
    else:
        fieldnames = ["path", "breed", "dataset_code", "source_type", "role",
                      "width", "height", "format", "mode", "is_grayscale",
                      "aspect_ratio", "resolution_category", "file_size_bytes", "status"]
    with open(MASTER_METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    # Step 5: rebuild final_15breed_manifest.csv from remaining valid rows
    manifest_rows = [
        {"path": r["path"], "breed": r["breed"], "role": r["role"]}
        for r in kept_rows
        if r["role"] != "discovered_unlisted" and r["status"] == "ok"
    ]
    with open(FINAL_15BREED_MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "breed", "role"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Summary
    breed_counts = defaultdict(int)
    for row in manifest_rows:
        breed_counts[row["breed"]] += 1

    print(f"\nMoved: {moved_count}, already-missing (skipped): {missing_count}")
    print(f"master_metadata.csv rewritten: {len(kept_rows)} rows remain")
    print(f"final_15breed_manifest.csv rebuilt: {len(manifest_rows)} rows")
    print("\nPer-breed counts after quarantine:")
    for breed, count in sorted(breed_counts.items(), key=lambda x: -x[1]):
        print(f"  {breed:22s} {count}")
    print(f"\nQuarantined files moved to: {UNEXPLAINED_QUARANTINE_DIR}")
    print(f"Quarantine log: {QUARANTINE_LOG_CSV}")


if __name__ == "__main__":
    main()
