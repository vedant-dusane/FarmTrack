# ---------------------------------------------------------------------------
# Example run:
# python 07_split_dataset.py
# ---------------------------------------------------------------------------

"""
07_split_dataset.py

Purpose (per roadmap step 8 — Train/Validation/Test Split)
---------------------------------------------------------------
Splits master_dataset/<breed>/ (flat folders) into:

    master_dataset/train/<breed>/
    master_dataset/val/<breed>/
    master_dataset/test/<breed>/

Design requirements (agreed through discussion)
--------------------------------------------------
1. CLUSTER-AWARE / leakage-safe: a duplicate cluster (canonical image +
   its augmented siblings, or a Kankrej top-up image + its source) is
   NEVER split across train/val/test — the whole cluster goes to one
   split only.
2. "More unique in val/test": small/singleton clusters (genuinely unique,
   non-duplicated images) are preferentially assigned to val/test first;
   larger clusters (dominated by synthetic augmented siblings) default to
   train. This maximizes how much of val/test is real, independent data.
3. GRAYSCALE-AWARE: stratification is done separately for grayscale vs
   color images WITHIN each breed, so each split gets a proportional
   share of that breed's grayscale content — prevents the model from
   learning "grayscale -> Holstein-Friesian" as a split-specific
   artifact rather than a real feature.
4. Default ratio 70/15/15 (adjustable). Physically MOVES files (not
   copies) — old flat breed folders are removed once empty, same
   safe-check pattern as 04b_finalize_breed_selection.py.

Cluster resolution
--------------------
- Canonical / augmented images: cluster_id looked up from
  duplicate_clusters.csv (04's output) via the 'new_path' column, which
  covers kept_unique, kept_canonical, and kept_augmented_source(->new
  augmented file) rows.
- Kankrej top-up images (04b's output, not tracked in duplicate_clusters.csv):
  resolved by parsing the '__topupN_i' suffix out of the filename to
  recover the source image's original stem, then merging the top-up
  image's cluster with its source's cluster so they can't be split apart.
  Falls back to treating it as its own singleton cluster if the source
  can't be resolved (safe default, tiny edge case — ~97 images).
- Anything else unresolved: singleton cluster (its own path as the key).

Output
------
- master_dataset/train|val|test/<breed>/ — physically moved files.
- split_manifest.csv: path, breed, split, role, is_grayscale, cluster_key.
- Console summary: per-breed per-split counts, grayscale % per split
  (should stay consistent across splits per breed), real/synthetic mix.

Usage
-----
1. Edit MASTER_METADATA_CSV, DUPLICATE_CLUSTERS_CSV, MASTER_DATASET_ROOT.
2. Adjust SPLIT_RATIOS if needed.
3. Run: python 07_split_dataset.py
"""

import csv
import random
import shutil
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

MASTER_METADATA_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\master_metadata.csv"
DUPLICATE_CLUSTERS_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\duplicate_clusters.csv"
MASTER_DATASET_ROOT = r"D:\Projects\Federated_Learning\Master-Dataset\master_dataset"

SPLIT_MANIFEST_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\split_manifest.csv"

SELECTED_BREEDS = [
    "holstein-friesian", "gir", "sahiwal", "jersey", "ayrshire",
    "brown-swiss", "jaffarabadi", "nagori", "nili-ravi", "tharparkar",
    "kankrej", "umblachery", "rathi", "red-sindhi", "vechur",
]

SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Cluster resolution
# ---------------------------------------------------------------------------

def build_cluster_map(master_rows, duplicate_clusters_csv):
    """
    Returns dict: path -> cluster_key
    Every path in master_rows gets an entry, falling back to a singleton
    (its own path) if no cluster info is found.
    """
    path_to_cluster = {}

    try:
        with open(duplicate_clusters_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                new_path = row.get("new_path")
                cluster_id = row.get("cluster_id")
                if new_path and cluster_id:
                    path_to_cluster[new_path] = cluster_id
    except FileNotFoundError:
        print("  [WARNING] duplicate_clusters.csv not found — proceeding with singletons only.")

    all_paths = {r["path"] for r in master_rows}

    # index paths by (breed, directory) for fast top-up source lookup
    paths_by_dir = defaultdict(set)
    for r in master_rows:
        p = Path(r["path"])
        paths_by_dir[str(p.parent)].add(p.name)

    unresolved = []
    for row in master_rows:
        path = row["path"]
        if path in path_to_cluster:
            continue

        name = Path(path).name
        if "__topup" in name:
            # recover source stem: everything before '__topup'
            source_stem_part = name.split("__topup", 1)[0]
            parent_dir = str(Path(path).parent)
            # find a sibling file whose name starts with that stem and
            # is NOT itself a topup/aug file (the real source original)
            candidate = None
            for sibling_name in paths_by_dir[parent_dir]:
                if sibling_name.startswith(source_stem_part) and "__topup" not in sibling_name and "__aug" not in sibling_name:
                    candidate = str(Path(parent_dir) / sibling_name)
                    break
            if candidate and candidate in all_paths:
                # merge: use source's existing cluster if it has one,
                # else create a shared new one
                source_cluster = path_to_cluster.get(candidate, f"topup_{candidate}")
                path_to_cluster[candidate] = source_cluster
                path_to_cluster[path] = source_cluster
                continue

        unresolved.append(path)

    for path in unresolved:
        path_to_cluster[path] = f"singleton_{path}"

    return path_to_cluster


# ---------------------------------------------------------------------------
# Stratified cluster-aware split
# ---------------------------------------------------------------------------

def assign_clusters_to_splits(clusters, ratios, rng):
    """
    clusters: dict cluster_key -> list of rows
    Returns dict cluster_key -> split name
    Greedy: process clusters smallest-first, assign each fully to whichever
    split has the largest REMAINING deficit as a fraction of its own
    target (keeps small/val/test targets from being starved by train's
    much larger absolute target).
    """
    total = sum(len(rows) for rows in clusters.values())
    if total == 0:
        return {}

    targets = {split: total * ratio for split, ratio in ratios.items()}
    remaining = dict(targets)

    cluster_items = list(clusters.items())
    rng.shuffle(cluster_items)  # tie-break randomness, reproducible via seed
    cluster_items.sort(key=lambda kv: len(kv[1]))  # smallest clusters first

    assignment = {}
    for cluster_key, rows in cluster_items:
        size = len(rows)
        best_split = max(
            remaining,
            key=lambda s: remaining[s] / targets[s] if targets[s] > 0 else -999
        )
        assignment[cluster_key] = best_split
        remaining[best_split] -= size

    return assignment


def main():
    rng = random.Random(RANDOM_SEED)

    with open(MASTER_METADATA_CSV, newline="", encoding="utf-8") as f:
        all_rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]

    print(f"Loaded {len(all_rows)} images from master_metadata.csv")
    print("Building cluster map (leakage-safe grouping)...")
    cluster_map = build_cluster_map(all_rows, DUPLICATE_CLUSTERS_CSV)

    # Group rows by (breed, is_grayscale) substratum, then by cluster
    substrata = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        key = (row["breed"], row["is_grayscale"])
        cluster_key = cluster_map[row["path"]]
        substrata[key][cluster_key].append(row)

    split_assignment_by_path = {}
    print("\nRunning stratified cluster-aware split per breed x grayscale group...")
    for (breed, is_gray), clusters in substrata.items():
        assignment = assign_clusters_to_splits(clusters, SPLIT_RATIOS, rng)
        for cluster_key, split in assignment.items():
            for row in clusters[cluster_key]:
                split_assignment_by_path[row["path"]] = split

    # --- Physically move files ---
    print("\nMoving files into train/val/test folders...")
    manifest_rows = []
    moved_count = 0
    for row in all_rows:
        old_path = Path(row["path"])
        split = split_assignment_by_path[row["path"]]
        new_dir = Path(MASTER_DATASET_ROOT) / split / row["breed"]
        new_dir.mkdir(parents=True, exist_ok=True)
        new_path = new_dir / old_path.name

        if old_path.exists():
            try:
                shutil.move(str(old_path), str(new_path))
                moved_count += 1
            except Exception as e:
                print(f"  [WARNING] failed to move {old_path}: {e}")
                continue
        else:
            print(f"  [WARNING] source missing, skipping: {old_path}")
            continue

        manifest_rows.append({
            "path": str(new_path), "breed": row["breed"], "split": split,
            "role": row["role"], "is_grayscale": row["is_grayscale"],
            "cluster_key": cluster_map[row["path"]],
        })

    with open(SPLIT_MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "breed", "split", "role", "is_grayscale", "cluster_key"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # --- Remove now-empty old flat breed folders ---
    print("\nCleaning up old flat breed folders...")
    removed = 0
    for breed in SELECTED_BREEDS:
        old_dir = Path(MASTER_DATASET_ROOT) / breed
        if old_dir.exists():
            remaining = list(old_dir.iterdir())
            if not remaining:
                old_dir.rmdir()
                removed += 1
            else:
                print(f"  [SKIPPED] {old_dir} still has {len(remaining)} file(s)")
    print(f"Removed {removed} empty flat breed folders.")

    # --- Summary ---
    breed_split_counts = defaultdict(lambda: defaultdict(int))
    breed_split_gray = defaultdict(lambda: defaultdict(int))
    for row in manifest_rows:
        breed_split_counts[row["breed"]][row["split"]] += 1
        if row["is_grayscale"] == "True":
            breed_split_gray[row["breed"]][row["split"]] += 1

    print("\n" + "=" * 80)
    print("SPLIT SUMMARY")
    print("=" * 80)
    print(f"{'breed':22s}{'train':>8s}{'val':>8s}{'test':>8s}{'total':>8s}   grayscale% (train/val/test)")
    for breed in SELECTED_BREEDS:
        counts = breed_split_counts[breed]
        gray = breed_split_gray[breed]
        total = sum(counts.values())
        tr, va, te = counts.get("train", 0), counts.get("val", 0), counts.get("test", 0)
        gtr = gray.get("train", 0) / tr * 100 if tr else 0
        gva = gray.get("val", 0) / va * 100 if va else 0
        gte = gray.get("test", 0) / te * 100 if te else 0
        print(f"{breed:22s}{tr:8d}{va:8d}{te:8d}{total:8d}   {gtr:.1f}% / {gva:.1f}% / {gte:.1f}%")

    print(f"\nTotal moved: {moved_count}")
    print(f"Split manifest: {SPLIT_MANIFEST_CSV}")


if __name__ == "__main__":
    main()
