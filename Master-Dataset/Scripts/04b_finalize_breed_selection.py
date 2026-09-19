# ---------------------------------------------------------------------------
# Example run:
# python 04b_finalize_breed_selection.py
#
# Requires: pip install pillow
# ---------------------------------------------------------------------------

# 04b_finalize_breed_selection.py

"""
Purpose
-------
Narrows the dataset down from 84 breeds to the 15 breeds selected for the
final training scope. For breeds NOT in the selected list, moves all their
images out of master_dataset/ into a discarded_breeds quarantine folder
(never deletes — fully reversible). For the 15 selected breeds, keeps
everything as-is EXCEPT any breed still below TOTAL_FLOOR after selection,
which gets topped up via self-augmentation of its own existing images
(only Kankrej currently needs this, generated fresh from its own unique
photos since it has zero existing duplicates to draw from).

No per-breed CEILING is applied — breeds with large totals (e.g. Sahiwal
3300, Murrah 1506) keep every image. The floor is a selection filter only,
not a cap. Class imbalance among the kept breeds is meant to be handled at
training time (class weighting), not by deleting real images here.

Output
------
- final_15breed_manifest.csv: the surviving image list (path, breed, role)
  for ONLY the 15 selected breeds, including any top-up augmented images.
  This is what 05_generate_metadata.py onward should read from.
- discarded_breeds_log.csv: audit trail of every image moved out, with
  its new location.
- Non-selected breeds' images moved to:
  D:\\Projects\\Federated_Learning\\Master-Dataset\\scrapped_photos\\discraded_breeds\\<breed>\\

Usage
-----
1. Edit FINAL_DATASET_MANIFEST_CSV (input, from 04_remove_duplicates.py).
2. Adjust SELECTED_BREEDS / TOTAL_FLOOR if the final breed list changes.
3. Run: python 04b_finalize_breed_selection.py
"""

import csv
import random
from pathlib import Path
import shutil

try:
    from PIL import Image, ImageEnhance
except ImportError:
    raise SystemExit("Missing dependency. Install with:\n    pip install pillow")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

FINAL_DATASET_MANIFEST_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\final_dataset_manifest.csv"
FINAL_15BREED_MANIFEST_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\final_15breed_manifest.csv"
DISCARDED_BREEDS_LOG_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\discarded_breeds_log.csv"

DISCARDED_BREEDS_DIR = r"D:\Projects\Federated_Learning\Master-Dataset\scrapped_photos\discraded_breeds"

SELECTED_BREEDS = [
    "holstein-friesian", "gir", "sahiwal", "jersey", "ayrshire",
    "brown-swiss", "jaffarabadi", "nagori", "nili-ravi", "tharparkar",
    "kankrej", "umblachery", "rathi", "red-sindhi", "vechur",
]

TOTAL_FLOOR = 600   # any selected breed below this gets topped up via self-augmentation


# ---------------------------------------------------------------------------
# Top-up augmentation (reused approach from 04_remove_duplicates.py)
# ---------------------------------------------------------------------------

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


def top_up_breed(breed, current_rows, deficit, rng):
    """
    Generates `deficit` new augmented images from this breed's own existing
    photos (cycling through them if deficit > available source count).
    Returns list of new manifest rows for the generated images.
    """
    # prefer 'unique' role sources first (freshest, no cluster complications)
    sources = [r for r in current_rows if r["role"] == "unique"] or current_rows
    new_rows = []

    for i in range(deficit):
        src_row = sources[i % len(sources)]
        src_path = Path(src_row["path"])
        variant_num = (i // len(sources)) + 1
        aug_name = f"{src_path.stem}__topup{variant_num}_{i}{src_path.suffix}"
        aug_dest = src_path.parent / aug_name
        try:
            augment_image(src_path, aug_dest, rng)
            new_rows.append({"path": str(aug_dest), "breed": breed, "role": "augmented_topup"})
        except Exception as e:
            print(f"  [WARNING] top-up augmentation failed for {src_path}: {e}")

    return new_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_rows = []
    with open(FINAL_DATASET_MANIFEST_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            all_rows.append(row)

    selected_set = set(SELECTED_BREEDS)
    kept_rows = [r for r in all_rows if r["breed"] in selected_set]
    discarded_rows = [r for r in all_rows if r["breed"] not in selected_set]

    print(f"Total images in input manifest: {len(all_rows)}")
    print(f"Kept (15 selected breeds): {len(kept_rows)}")
    print(f"Discarded (not in selection): {len(discarded_rows)}\n")

    # --- Move discarded breeds' images to quarantine ---
    discarded_log_rows = []
    print("Moving non-selected breed images to discarded_breeds quarantine...")
    for row in discarded_rows:
        src_path = Path(row["path"])
        breed = row["breed"]
        dest_dir = Path(DISCARDED_BREEDS_DIR) / breed
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / src_path.name
        moved = False
        if src_path.exists():
            try:
                shutil.move(str(src_path), str(dest_path))
                moved = True
            except Exception as e:
                print(f"  [WARNING] failed to move {src_path}: {e}")
        discarded_log_rows.append({
            "breed": breed, "original_path": row["path"],
            "new_path": str(dest_path) if moved else "",
            "moved": moved,
        })

    with open(DISCARDED_BREEDS_LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["breed", "original_path", "new_path", "moved"])
        writer.writeheader()
        writer.writerows(discarded_log_rows)

    # --- Remove now-empty breed folders for discarded breeds ---
    print("\nRemoving emptied breed folders from master_dataset...")
    discarded_folders = set()
    for row in discarded_rows:
        discarded_folders.add(Path(row["path"]).parent)

    removed_count = 0
    kept_nonempty_count = 0
    for folder in sorted(discarded_folders):
        if not folder.exists():
            continue
        remaining = list(folder.iterdir())
        if not remaining:
            try:
                folder.rmdir()
                removed_count += 1
            except Exception as e:
                print(f"  [WARNING] could not remove {folder}: {e}")
        else:
            kept_nonempty_count += 1
            print(f"  [SKIPPED] {folder} still has {len(remaining)} file(s) — not removed "
                  f"(some moves may have failed, check discarded_breeds_log.csv)")

    print(f"Removed {removed_count} empty breed folders.")
    if kept_nonempty_count:
        print(f"{kept_nonempty_count} folder(s) left in place because they weren't fully empty — see warnings above.")

    # --- Check totals for selected breeds, top up any below TOTAL_FLOOR ---
    rng = random.Random(42)
    rows_by_breed = {}
    for breed in SELECTED_BREEDS:
        rows_by_breed[breed] = [r for r in kept_rows if r["breed"] == breed]

    print("\nChecking selected breeds against TOTAL_FLOOR...")
    final_rows = list(kept_rows)
    for breed in SELECTED_BREEDS:
        current_count = len(rows_by_breed[breed])
        if current_count < TOTAL_FLOOR:
            deficit = TOTAL_FLOOR - current_count
            print(f"  {breed}: {current_count} < {TOTAL_FLOOR} floor, generating {deficit} top-up images...")
            new_rows = top_up_breed(breed, rows_by_breed[breed], deficit, rng)
            final_rows.extend(new_rows)
            print(f"    -> {breed} now at {current_count + len(new_rows)}")
        else:
            print(f"  {breed}: {current_count} (OK, no top-up needed)")

    with open(FINAL_15BREED_MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "breed", "role"])
        writer.writeheader()
        writer.writerows(final_rows)

    # --- Summary ---
    from collections import defaultdict
    final_breed_counts = defaultdict(int)
    for row in final_rows:
        final_breed_counts[row["breed"]] += 1

    print("\n" + "=" * 60)
    print("FINAL 15-BREED DATASET SUMMARY")
    print("=" * 60)
    grand_total = 0
    for breed in SELECTED_BREEDS:
        count = final_breed_counts.get(breed, 0)
        grand_total += count
        print(f"{breed:25s} {count}")
    print("-" * 60)
    print(f"{'TOTAL':25s} {grand_total}")
    print(f"\nFinal manifest: {FINAL_15BREED_MANIFEST_CSV}")
    print(f"Discarded breeds moved to: {DISCARDED_BREEDS_DIR}")
    print(f"Discarded log: {DISCARDED_BREEDS_LOG_CSV}")


if __name__ == "__main__":
    main()