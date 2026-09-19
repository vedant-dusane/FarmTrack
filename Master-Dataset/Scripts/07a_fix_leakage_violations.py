# ---------------------------------------------------------------------------
# Example run:
# python 07a_fix_leakage_violations.py
# ---------------------------------------------------------------------------

"""
07a_fix_leakage_violations.py

Purpose
-------
Repairs the 48 leakage violations found in split_manifest.csv (from
07_split_dataset.py), where a duplicate cluster ended up split across
more than one of train/val/test. Root cause: stratifying by
(breed, is_grayscale) at the ROW level meant a canonical image and one
of its own augmented siblings could occasionally disagree on
is_grayscale (augmentation's brightness/crop/rotation can nudge a
borderline image across the grayscale threshold), sorting them into
different grayscale substrata that were then split independently.

This script does NOT re-run the split from scratch — it directly repairs
the existing split_manifest.csv and physically relocates only the
affected files.

Method
------
1. Group split_manifest.csv rows by cluster_key.
2. Find clusters whose members currently sit in more than one split.
3. For each such cluster, pick a TARGET split = majority vote among its
   current members' splits (tie-broken toward 'train', the safest
   default for an ambiguous mixed cluster).
4. Physically move only the MINORITY-split members' files into the
   target split's matching breed folder.
5. Rewrite split_manifest.csv with corrected paths/splits.

Output
------
- split_manifest.csv: updated in place (old version backed up first).
- leakage_fix_report.csv: exactly which files moved, from which split
  to which, and why (cluster majority vote).

Usage
-----
1. Edit SPLIT_MANIFEST_CSV / MASTER_DATASET_ROOT below.
2. Run: python 07a_fix_leakage_violations.py
3. Confirm the printed violation count is 0 after the fix.
"""

import csv
import shutil
from pathlib import Path
from collections import defaultdict, Counter

SPLIT_MANIFEST_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\split_manifest.csv"
MANIFEST_BACKUP_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\split_manifest_backup.csv"
LEAKAGE_FIX_REPORT_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\leakage_fix_report.csv"
MASTER_DATASET_ROOT = r"D:\Projects\Federated_Learning\Master-Dataset\master_dataset"


def main():
    with open(SPLIT_MANIFEST_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    shutil.copy(SPLIT_MANIFEST_CSV, MANIFEST_BACKUP_CSV)
    print(f"Backed up split_manifest.csv to: {MANIFEST_BACKUP_CSV}\n")

    clusters = defaultdict(list)
    for row in rows:
        clusters[row["cluster_key"]].append(row)

    violations = {k: v for k, v in clusters.items() if len({r["split"] for r in v}) > 1}
    print(f"Total clusters: {len(clusters)}")
    print(f"Violating clusters found: {len(violations)}\n")

    fix_report_rows = []
    moved_count = 0

    for cluster_key, members in violations.items():
        split_counts = Counter(r["split"] for r in members)
        max_count = max(split_counts.values())
        candidates = [s for s, c in split_counts.items() if c == max_count]
        target_split = "train" if "train" in candidates else sorted(candidates)[0]

        for row in members:
            if row["split"] == target_split:
                continue  # already in the target split, no move needed

            old_path = Path(row["path"])
            breed = row["breed"]
            new_dir = Path(MASTER_DATASET_ROOT) / target_split / breed
            new_dir.mkdir(parents=True, exist_ok=True)
            new_path = new_dir / old_path.name

            moved = False
            if old_path.exists():
                try:
                    shutil.move(str(old_path), str(new_path))
                    moved = True
                    moved_count += 1
                except Exception as e:
                    print(f"  [WARNING] failed to move {old_path}: {e}")
            else:
                print(f"  [WARNING] source missing, skipping: {old_path}")

            fix_report_rows.append({
                "cluster_key": cluster_key, "breed": breed,
                "old_split": row["split"], "new_split": target_split,
                "old_path": str(old_path), "new_path": str(new_path) if moved else "",
                "moved": moved,
            })

            if moved:
                row["split"] = target_split
                row["path"] = str(new_path)

    with open(SPLIT_MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "breed", "split", "role", "is_grayscale", "cluster_key"])
        writer.writeheader()
        writer.writerows(rows)

    with open(LEAKAGE_FIX_REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "cluster_key", "breed", "old_split", "new_split", "old_path", "new_path", "moved"
        ])
        writer.writeheader()
        writer.writerows(fix_report_rows)

    # Verify
    clusters_after = defaultdict(set)
    for row in rows:
        clusters_after[row["cluster_key"]].add(row["split"])
    remaining_violations = sum(1 for v in clusters_after.values() if len(v) > 1)

    print(f"Files moved: {moved_count}")
    print(f"Remaining leakage violations after fix: {remaining_violations}")
    print(f"\nUpdated manifest: {SPLIT_MANIFEST_CSV}")
    print(f"Fix report: {LEAKAGE_FIX_REPORT_CSV}")


if __name__ == "__main__":
    main()
