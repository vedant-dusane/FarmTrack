# ---------------------------------------------------------------------------
# Example run:
# python 04_remove_duplicates.py
#
# Requires: pip install imagehash pillow opencv-python-headless numpy
# ---------------------------------------------------------------------------

"""

Purpose (per roadmap step 5 — Duplicate Removal)
---------------------------------------------------
Finds duplicate images WITHIN each breed folder using perceptual hashing
(including horizontal-flip detection), then:

    TIGHT duplicates (near-identical, hash distance <= TIGHT_THRESHOLD):
        - Keep the sharpest+highest-resolution copy as canonical.
        - Augment up to AUGMENT_COUNT of the rest into genuinely new
          datapoints (flip/rotate/brightness/zoom).
        - Anything beyond that cap is QUARANTINED (moved, never deleted)
          to master_dataset/scrapped_photos/.

    LOOSE duplicates (distance between TIGHT_THRESHOLD and LOOSE_THRESHOLD):
        - Only LOGGED for review. Left completely untouched — these likely
          represent a real (if similar) distinct photo, not a copy.

Also detects EXACT cross-breed hash collisions (same photo appearing under
two different breed labels) — reported only, never auto-resolved, since
fixing a mislabel needs a human to look at the actual photo.

Canonical selection
--------------------
Within a tight cluster: images whose sharpness (Laplacian variance) is
below SHARPNESS_FLOOR_RATIO of the cluster's max sharpness are excluded
from canonical consideration (too blurry to be the "best" copy) UNLESS
that would exclude every candidate, in which case the sharpness filter is
skipped and resolution alone decides. Among remaining candidates, highest
resolution (width*height) wins.

Flip detection
----------------
Each image gets a normal phash AND a phash of its horizontally-flipped
version. Clustering distance between two images = min(normal-vs-normal,
flipped-vs-normal) — catches mirrored duplicates from untrusted sources
without needing to compute a second flipped hash for both sides.

Does NOT touch your original 11 raw datasets — only operates on files
already copied into master_dataset/.

Output
------
- duplicate_clusters.csv: full audit trail per image (breed, dataset_code,
  dataset_full_name, cluster_id, duplicate_type, role, new_path).
- loose_duplicates_log.csv: pairs flagged as loose/similar-but-kept,
  for manual review only.
- cross_breed_exact_collisions.csv: exact-hash matches found across
  DIFFERENT breed folders — likely mislabeling, needs manual review.
- final_dataset_manifest.csv: final list of surviving image paths for
  05_generate_metadata.py onward.
- Quarantined files moved to master_dataset/scrapped_photos/<reason>/<breed>/.

Usage
-----
1. Edit the CONFIG paths below.
2. Run: python 04_remove_duplicates.py
3. Review console summary + the three CSVs before moving to 05.
"""

import csv
import random
import shutil
from pathlib import Path
from collections import defaultdict

try:
    from PIL import Image, ImageEnhance
    import imagehash
    import cv2
    import numpy as np
except ImportError:
    raise SystemExit(
        "Missing dependencies. Install with:\n"
        "    pip install imagehash pillow opencv-python-headless numpy"
    )

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

METADATA_CSV_PATH = r"D:\Projects\Federated_Learning\Master-Dataset\metadata.csv"
VALIDATION_REPORT_CSV_PATH = r"D:\Projects\Federated_Learning\Master-Dataset\validation_report.csv"

MASTER_DATASET_ROOT = r"D:\Projects\Federated_Learning\Master-Dataset"
SCRAPPED_DIR_NAME = "scrapped_photos"   # created INSIDE MASTER_DATASET_ROOT

DUPLICATE_CLUSTERS_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\duplicate_clusters.csv"
LOOSE_DUPLICATES_LOG_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\loose_duplicates_log.csv"
CROSS_BREED_COLLISIONS_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\cross_breed_exact_collisions.csv"
FINAL_MANIFEST_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\final_dataset_manifest.csv"

HASH_SIZE = 8                    # standard phash size (64-bit)
TIGHT_THRESHOLD = 4              # <= this: treated as a hard duplicate (canonical+augment+drop)
LOOSE_THRESHOLD = 8              # between TIGHT and this: logged only, untouched

AUGMENT_COUNT = 6                # max duplicates per tight cluster turned into augmented images
SHARPNESS_FLOOR_RATIO = 0.3      # candidates below 30% of cluster's max sharpness are excluded
                                  # from canonical consideration (too blurry), unless that
                                  # would exclude everyone

INVALID_STATUSES = {"too_small", "corrupt", "unreadable"}


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# ---------------------------------------------------------------------------
# Hashing / sharpness helpers
# ---------------------------------------------------------------------------

def compute_hashes(path: Path):
    """Returns (normal_hash, flipped_hash) or (None, None) if unreadable."""
    try:
        with Image.open(path) as img:
            gray = img.convert("L")
            normal_hash = imagehash.phash(gray, hash_size=HASH_SIZE)
            flipped = gray.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_hash = imagehash.phash(flipped, hash_size=HASH_SIZE)
            return normal_hash, flipped_hash
    except Exception:
        return None, None


def min_distance(hashes_a, hashes_b):
    """hashes_* = (normal, flipped). Returns min distance across normal-normal
    and flipped_a-normal_b (sufficient to catch mirror duplicates either way)."""
    normal_a, flipped_a = hashes_a
    normal_b, _ = hashes_b
    return min(normal_a - normal_b, flipped_a - normal_b)


def compute_sharpness(path: Path):
    """Laplacian variance — higher = sharper. Returns 0.0 on failure."""
    try:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except Exception:
        return 0.0


def augment_image(src_path: Path, dest_path: Path, rng: random.Random):
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        if rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        angle = rng.uniform(-15, 15)
        img = img.rotate(angle, expand=False, fillcolor=(0, 0, 0))
        brightness_factor = rng.uniform(0.85, 1.15)
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)
        w, h = img.size
        zoom_frac = rng.uniform(0.85, 0.95)
        crop_w, crop_h = int(w * zoom_frac), int(h * zoom_frac)
        left = rng.randint(0, max(0, w - crop_w))
        top = rng.randint(0, max(0, h - crop_h))
        img = img.crop((left, top, left + crop_w, top + crop_h)).resize((w, h))
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest_path, quality=95)


def scrap_file(path: Path, breed: str, reason: str):
    dest_dir = Path(MASTER_DATASET_ROOT) / SCRAPPED_DIR_NAME / reason / breed
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    if path.exists():
        shutil.move(str(path), str(dest))
    return str(dest)


# ---------------------------------------------------------------------------
# Core per-breed dedup logic
# ---------------------------------------------------------------------------

def process_breed(breed, image_paths, dims_lookup, source_info, report_rows,
                   manifest_rows, loose_rows, rng):
    hashes = {}
    for path_str in image_paths:
        path = Path(path_str)
        h = compute_hashes(path)
        if h[0] is None:
            new_path = scrap_file(path, breed, "invalid_unreadable_at_dedup")
            _log_row(report_rows, breed, path_str, "", "", "rejected_invalid", new_path, source_info)
            continue
        hashes[path_str] = h

    if not hashes:
        return

    paths_list = list(hashes.keys())
    n = len(paths_list)
    uf = UnionFind(paths_list)

    for i in range(n):
        for j in range(i + 1, n):
            pa, pb = paths_list[i], paths_list[j]
            d = min_distance(hashes[pa], hashes[pb])
            if d <= TIGHT_THRESHOLD:
                uf.union(pa, pb)
            elif d <= LOOSE_THRESHOLD:
                loose_rows.append({"breed": breed, "path_a": pa, "path_b": pb, "distance": d})

    clusters = defaultdict(list)
    for path_str in paths_list:
        clusters[uf.find(path_str)].append(path_str)

    for cluster_idx, (root, paths) in enumerate(clusters.items(), start=1):
        cluster_id = f"{breed}_{cluster_idx}"

        if len(paths) == 1:
            path_str = paths[0]
            _log_row(report_rows, breed, path_str, cluster_id, "unique", "kept_unique", path_str, source_info)
            manifest_rows.append({"path": path_str, "breed": breed, "role": "unique"})
            continue

        sharpness = {p: compute_sharpness(Path(p)) for p in paths}
        max_sharp = max(sharpness.values()) if sharpness else 0.0
        if max_sharp > 0:
            candidates = [p for p in paths if sharpness[p] >= SHARPNESS_FLOOR_RATIO * max_sharp]
        else:
            candidates = paths
        if not candidates:
            candidates = paths

        def area(p):
            dims = dims_lookup.get(p)
            return (dims[0] * dims[1]) if dims else 0

        canonical = max(candidates, key=area)
        remaining = [p for p in paths if p != canonical]

        _log_row(report_rows, breed, canonical, cluster_id, "tight", "kept_canonical", canonical, source_info)
        manifest_rows.append({"path": canonical, "breed": breed, "role": "canonical"})

        to_augment = remaining[:AUGMENT_COUNT]
        to_drop = remaining[AUGMENT_COUNT:]

        for i, dup_path_str in enumerate(to_augment, start=1):
            dup_path = Path(dup_path_str)
            aug_name = f"{dup_path.stem}__aug{i}_{cluster_id}{dup_path.suffix}"
            aug_dest = dup_path.parent / aug_name
            try:
                augment_image(dup_path, aug_dest, rng)
                scrap_file(dup_path, breed, "duplicates_augmented_source")
                _log_row(report_rows, breed, dup_path_str, cluster_id, "tight",
                          "kept_augmented_source", str(aug_dest), source_info)
                manifest_rows.append({"path": str(aug_dest), "breed": breed, "role": "augmented"})
            except Exception as e:
                _log_row(report_rows, breed, dup_path_str, cluster_id, "tight",
                          f"augment_failed: {e}", "", source_info)

        for dup_path_str in to_drop:
            dup_path = Path(dup_path_str)
            new_path = scrap_file(dup_path, breed, "duplicates_dropped")
            _log_row(report_rows, breed, dup_path_str, cluster_id, "tight", "dropped", new_path, source_info)


def _log_row(report_rows, breed, path_str, cluster_id, dup_type, role, new_path, source_info):
    info = source_info.get(path_str, {})
    report_rows.append({
        "breed": breed,
        "dataset_code": info.get("dataset_code", ""),
        "dataset_full_name": info.get("dataset_full_name", ""),
        "original_path": path_str,
        "cluster_id": cluster_id,
        "duplicate_type": dup_type,
        "role": role,
        "new_path": new_path,
    })


# ---------------------------------------------------------------------------
# Cross-breed EXACT collision detection
# ---------------------------------------------------------------------------

def find_cross_breed_exact_collisions(all_hashes_by_path, breed_by_path):
    by_hash = defaultdict(list)
    for path_str, (normal_h, _) in all_hashes_by_path.items():
        by_hash[str(normal_h)].append(path_str)

    collisions = []
    for hash_str, paths in by_hash.items():
        breeds_involved = set(breed_by_path[p] for p in paths)
        if len(breeds_involved) > 1:
            for p in paths:
                collisions.append({"hash": hash_str, "path": p, "breed": breed_by_path[p]})
    return collisions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = random.Random(42)

    breed_by_path = {}
    source_info = {}
    with open(METADATA_CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            breed_by_path[row["master_path"]] = row["breed"]
            source_info[row["master_path"]] = {
                "dataset_code": row.get("dataset_code", ""),
                "dataset_full_name": row.get("dataset_full_name", ""),
            }

    status_by_path = {}
    dims_lookup = {}
    with open(VALIDATION_REPORT_CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            status_by_path[row["master_path"]] = row["status"]
            if row["width"] and row["height"]:
                dims_lookup[row["master_path"]] = (int(row["width"]), int(row["height"]))

    report_rows = []
    manifest_rows = []
    loose_rows = []

    paths_by_breed = defaultdict(list)
    invalid_count = 0
    for path_str, breed in breed_by_path.items():
        status = status_by_path.get(path_str, "ok")
        if status in INVALID_STATUSES:
            path = Path(path_str)
            new_path = scrap_file(path, breed, "invalid_from_validation")
            _log_row(report_rows, breed, path_str, "", "", f"rejected_invalid ({status})", new_path, source_info)
            invalid_count += 1
        else:
            paths_by_breed[breed].append(path_str)

    print(f"Quarantined {invalid_count} invalid images (too_small/corrupt/unreadable).\n")
    print("Scanning for duplicates within each breed (tight+loose, flip-aware)...\n")

    all_hashes_by_path = {}

    for breed, paths in sorted(paths_by_breed.items()):
        print(f"Processing breed: {breed} ({len(paths)} images)")

        local_hashes = {}
        for path_str in paths:
            h = compute_hashes(Path(path_str))
            if h[0] is not None:
                local_hashes[path_str] = h
        all_hashes_by_path.update(local_hashes)

        process_breed(breed, paths, dims_lookup, source_info, report_rows,
                       manifest_rows, loose_rows, rng)

    print("\nChecking for cross-breed exact-hash collisions (possible mislabeling)...")
    collisions = find_cross_breed_exact_collisions(all_hashes_by_path, breed_by_path)

    with open(DUPLICATE_CLUSTERS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "breed", "dataset_code", "dataset_full_name", "original_path",
            "cluster_id", "duplicate_type", "role", "new_path"
        ])
        writer.writeheader()
        writer.writerows(report_rows)

    with open(LOOSE_DUPLICATES_LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["breed", "path_a", "path_b", "distance"])
        writer.writeheader()
        writer.writerows(loose_rows)

    with open(CROSS_BREED_COLLISIONS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["hash", "path", "breed"])
        writer.writeheader()
        writer.writerows(collisions)

    with open(FINAL_MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "breed", "role"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    role_counts = defaultdict(int)
    breed_final_counts = defaultdict(int)
    for row in report_rows:
        role_counts[row["role"].split(" (")[0]] += 1
    for row in manifest_rows:
        breed_final_counts[row["breed"]] += 1

    print("\n" + "=" * 70)
    print("DEDUPLICATION SUMMARY")
    print("=" * 70)
    for role, count in sorted(role_counts.items()):
        print(f"{role:35s} {count}")
    print("-" * 70)
    print(f"{'TOTAL IN final_dataset_manifest.csv':35s} {len(manifest_rows)}")
    print(f"Loose (similar-but-kept) pairs logged: {len(loose_rows)}")
    print(f"Cross-breed exact collisions found: {len(collisions)} rows "
          f"({'REVIEW NEEDED' if collisions else 'none'})")
    print(f"\nFull audit trail: {DUPLICATE_CLUSTERS_CSV}")
    print(f"Loose duplicates log: {LOOSE_DUPLICATES_LOG_CSV}")
    print(f"Cross-breed collisions: {CROSS_BREED_COLLISIONS_CSV}")
    print(f"Final manifest for step 05 onward: {FINAL_MANIFEST_CSV}")
    print(f"Quarantined files (reversible, not deleted): "
          f"{Path(MASTER_DATASET_ROOT) / SCRAPPED_DIR_NAME}")


if __name__ == "__main__":
    main()
