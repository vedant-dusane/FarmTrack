# ---------------------------------------------------------------------------
# Example run:
# python 04a_verify_cross_breed_collisions.py
#
# Requires: pip install opencv-python-headless scikit-image numpy
# ---------------------------------------------------------------------------

#04a_verify_cross_breed_collisions.py
"""
Purpose
-------
Takes the cross_breed_exact_collisions.csv produced by 04_remove_duplicates.py
(images whose perceptual hash matched EXACTLY across two different breed
folders) and determines, using actual pixel evidence rather than filenames
or hash alone, whether each case is:

    - a genuine duplicate/mislabel (same real photo under two breed labels), or
    - a coincidental hash collision between two visually similar but
      genuinely different animals (e.g. closely related buffalo breeds).

Only operates on the small set of ALREADY-FLAGGED collision images from
step 04 — not the full 48k dataset — so this stays fast despite doing
heavier per-pair checks.

Method (two tiers, cheapest/most certain first)
--------------------------------------------------
1. MD5 of raw file bytes: if two files across different breeds are
   byte-for-byte identical, that's certain proof of the same file being
   present under two labels. No judgment call needed.
2. SSIM (Structural Similarity Index) on actual pixel content (grayscale,
   resized to a common size) for anything not byte-identical. High SSIM
   (>= SSIM_HIGH_THRESHOLD) strongly suggests the same source photo
   (recompressed/resized copy). Low SSIM means genuinely different photos
   that just happened to produce the same coarse perceptual hash.

Does NOT move, rename, or relabel any files. Produces a review CSV only —
actual reassignment should be a separate, human-approved step.

Output
------
cross_breed_review.csv, one row per cross-breed pair within each collision
group:
    hash, breed_a, path_a, breed_b, path_b, md5_match, ssim_score, tier

tier is one of: byte_identical, high_ssim_match, low_ssim_different

Usage
-----
1. Edit CROSS_BREED_COLLISIONS_CSV (input from 04) and
   CROSS_BREED_REVIEW_CSV (output) paths below.
2. Adjust SSIM_HIGH_THRESHOLD if needed.
3. Run: python 04a_verify_cross_breed_collisions.py
4. Review cross_breed_review.csv, focusing on 'byte_identical' and
   'high_ssim_match' rows first. 'low_ssim_different' rows need no action.
"""

import csv
import hashlib
from pathlib import Path
from collections import defaultdict

try:
    import cv2
    import numpy as np
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    raise SystemExit(
        "Missing dependencies. Install with:\n"
        "    pip install opencv-python-headless scikit-image numpy"
    )

CROSS_BREED_COLLISIONS_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\cross_breed_exact_collisions.csv"
CROSS_BREED_REVIEW_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\cross_breed_review.csv"

SSIM_HIGH_THRESHOLD = 0.95
SSIM_COMPARE_SIZE = (256, 256)


def compute_md5(path: Path):
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def compute_ssim(path_a: Path, path_b: Path):
    try:
        img_a = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
        img_b = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)
        if img_a is None or img_b is None:
            return None
        img_a = cv2.resize(img_a, SSIM_COMPARE_SIZE)
        img_b = cv2.resize(img_b, SSIM_COMPARE_SIZE)
        score, _ = ssim(img_a, img_b, full=True)
        return float(score)
    except Exception:
        return None


def main():
    groups = defaultdict(list)
    with open(CROSS_BREED_COLLISIONS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            groups[row["hash"]].append((row["breed"], row["path"]))

    review_rows = []
    md5_cache = {}

    print(f"Verifying {len(groups)} collision groups with MD5 + SSIM...\n")

    for hash_val, members in groups.items():
        n = len(members)
        for i in range(n):
            for j in range(i + 1, n):
                breed_a, path_a = members[i]
                breed_b, path_b = members[j]
                if breed_a == breed_b:
                    continue

                pa, pb = Path(path_a), Path(path_b)

                if path_a not in md5_cache:
                    md5_cache[path_a] = compute_md5(pa)
                if path_b not in md5_cache:
                    md5_cache[path_b] = compute_md5(pb)

                md5_a, md5_b = md5_cache[path_a], md5_cache[path_b]
                md5_match = (md5_a is not None and md5_a == md5_b)

                if md5_match:
                    review_rows.append({
                        "hash": hash_val, "breed_a": breed_a, "path_a": path_a,
                        "breed_b": breed_b, "path_b": path_b,
                        "md5_match": True, "ssim_score": 1.0,
                        "tier": "byte_identical",
                    })
                    continue

                score = compute_ssim(pa, pb)
                if score is None:
                    tier = "comparison_failed"
                elif score >= SSIM_HIGH_THRESHOLD:
                    tier = "high_ssim_match"
                else:
                    tier = "low_ssim_different"

                review_rows.append({
                    "hash": hash_val, "breed_a": breed_a, "path_a": path_a,
                    "breed_b": breed_b, "path_b": path_b,
                    "md5_match": False, "ssim_score": score, "tier": tier,
                })

    with open(CROSS_BREED_REVIEW_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "hash", "breed_a", "path_a", "breed_b", "path_b",
            "md5_match", "ssim_score", "tier"
        ])
        writer.writeheader()
        writer.writerows(review_rows)

    tier_counts = defaultdict(int)
    tier_pairs = defaultdict(lambda: defaultdict(int))
    for row in review_rows:
        tier_counts[row["tier"]] += 1
        pair = tuple(sorted([row["breed_a"], row["breed_b"]]))
        tier_pairs[row["tier"]][pair] += 1

    print("=" * 70)
    print("CROSS-BREED VERIFICATION SUMMARY")
    print("=" * 70)
    for tier, count in sorted(tier_counts.items()):
        print(f"{tier:25s} {count}")
    print("-" * 70)
    print(f"{'TOTAL PAIRS CHECKED':25s} {len(review_rows)}")

    for tier in ("byte_identical", "high_ssim_match"):
        if tier_pairs[tier]:
            print(f"\nTop breed pairs needing review ({tier}):")
            top = sorted(tier_pairs[tier].items(), key=lambda x: -x[1])[:10]
            for pair, count in top:
                print(f"  {pair[0]} <-> {pair[1]}: {count}")

    print(f"\nFull review file: {CROSS_BREED_REVIEW_CSV}")
    print("Action needed on 'byte_identical' and 'high_ssim_match' rows only.")
    print("'low_ssim_different' rows are visually-similar-but-distinct breeds — no action needed.")


if __name__ == "__main__":
    main()
