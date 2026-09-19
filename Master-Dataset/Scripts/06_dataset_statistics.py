# ---------------------------------------------------------------------------
# Example run:
# python 06_dataset_statistics.py
#
# Requires: pip install matplotlib
# ---------------------------------------------------------------------------

"""
06_dataset_statistics.py

Purpose (per roadmap step 7 — Dataset Statistics)
------------------------------------------------------
Reads master_metadata.csv (from 05_generate_metadata.py — no folder
re-scanning here, that's already done) and produces a full statistical
picture of the final 15-breed dataset:

Base roadmap requirements:
    - Breed distribution
    - Dataset contribution (per source dataset, per breed)
    - Class imbalance report
    - Summary charts

Extended checks (beyond the base roadmap, agreed as worth adding):
    - Source-concentration risk: flags breeds overly reliant on a single
      source dataset (risk of the model learning dataset artifacts
      instead of real breed features).
    - Resolution consistency: flags breeds skewed toward smaller images
      vs. the rest of the dataset.
    - Grayscale concentration: verifies grayscale (CV-style) images stay
      a small % overall and aren't concentrated in any one breed.
    - Real vs. synthetic ratio per breed: how much of each breed is
      genuine (unique/canonical) vs. augmented/topup — feeds directly
      into 07_split_dataset.py's val/test-safe pool planning.

Output
------
- dataset_statistics.json: full machine-readable stats (matches the
  filename in your roadmap's final deliverable structure).
- dataset_health_report.md: human-readable narrative, flags anything
  concerning — usable directly in a paper's limitations section.
- 5 PNG charts in the charts/ subfolder.

Usage
-----
1. Edit MASTER_METADATA_CSV / OUTPUT_DIR below.
2. Run: python 06_dataset_statistics.py
3. Review dataset_health_report.md first — it summarizes anything
   that needs attention before moving to 07_split_dataset.py.
"""

import csv
import json
from pathlib import Path
from collections import defaultdict, Counter

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("Missing dependency. Install with:\n    pip install matplotlib")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

MASTER_METADATA_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\master_metadata.csv"
OUTPUT_DIR = r"D:\Projects\Federated_Learning\Master-Dataset\statistics"

# Thresholds for flagging concerns (adjustable)
SOURCE_CONCENTRATION_THRESHOLD = 0.60   # >60% of a breed from one dataset = flagged
RESOLUTION_SMALL_SKEW_MULTIPLIER = 1.5  # breed's 'small' % > 1.5x dataset average = flagged
GRAYSCALE_SKEW_MULTIPLIER = 2.0         # breed's grayscale % > 2x dataset average = flagged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = Path(OUTPUT_DIR)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(MASTER_METADATA_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["status"] == "ok":
                rows.append(row)

    total_images = len(rows)
    breeds = sorted(set(r["breed"] for r in rows))

    # --- Breed distribution ---
    breed_counts = Counter(r["breed"] for r in rows)

    # --- Dataset contribution (overall + per breed) ---
    dataset_counts_overall = Counter(r["dataset_code"] for r in rows)
    breed_dataset_matrix = defaultdict(lambda: Counter())
    for r in rows:
        breed_dataset_matrix[r["breed"]][r["dataset_code"]] += 1

    # --- Class imbalance ---
    max_breed = max(breed_counts, key=breed_counts.get)
    min_breed = min(breed_counts, key=breed_counts.get)
    imbalance_ratio = breed_counts[max_breed] / breed_counts[min_breed]
    mean_count = total_images / len(breeds)

    # --- Source concentration risk ---
    source_concentration_flags = {}
    for breed in breeds:
        dcounts = breed_dataset_matrix[breed]
        breed_total = sum(dcounts.values())
        top_dataset, top_count = dcounts.most_common(1)[0]
        share = top_count / breed_total
        if share >= SOURCE_CONCENTRATION_THRESHOLD:
            source_concentration_flags[breed] = {
                "dominant_dataset": top_dataset, "share": round(share, 3)
            }

    # --- Resolution consistency ---
    breed_resolution = defaultdict(lambda: Counter())
    for r in rows:
        breed_resolution[r["breed"]][r["resolution_category"]] += 1

    overall_small_pct = sum(1 for r in rows if r["resolution_category"] == "small") / total_images

    resolution_skew_flags = {}
    for breed in breeds:
        rc = breed_resolution[breed]
        breed_total = sum(rc.values())
        small_pct = rc.get("small", 0) / breed_total
        if overall_small_pct > 0 and small_pct >= overall_small_pct * RESOLUTION_SMALL_SKEW_MULTIPLIER and small_pct > 0.05:
            resolution_skew_flags[breed] = {
                "small_pct": round(small_pct, 3), "dataset_avg_small_pct": round(overall_small_pct, 3)
            }

    # --- Grayscale concentration ---
    overall_grayscale_pct = sum(1 for r in rows if r["is_grayscale"] == "True") / total_images
    breed_grayscale_pct = {}
    grayscale_flags = {}
    for breed in breeds:
        breed_rows = [r for r in rows if r["breed"] == breed]
        gpct = sum(1 for r in breed_rows if r["is_grayscale"] == "True") / len(breed_rows)
        breed_grayscale_pct[breed] = round(gpct, 3)
        if overall_grayscale_pct > 0 and gpct >= overall_grayscale_pct * GRAYSCALE_SKEW_MULTIPLIER and gpct > 0.02:
            grayscale_flags[breed] = {
                "grayscale_pct": round(gpct, 3), "dataset_avg_grayscale_pct": round(overall_grayscale_pct, 3)
            }

    # --- Real vs synthetic ratio per breed ---
    real_roles = {"unique", "canonical"}
    synthetic_roles = {"augmented", "augmented_topup"}
    real_synth_ratio = {}
    for breed in breeds:
        breed_rows = [r for r in rows if r["breed"] == breed]
        real_count = sum(1 for r in breed_rows if r["role"] in real_roles)
        synth_count = sum(1 for r in breed_rows if r["role"] in synthetic_roles)
        real_synth_ratio[breed] = {
            "real": real_count, "synthetic": synth_count,
            "real_pct": round(real_count / len(breed_rows), 3) if breed_rows else 0,
        }

    # -----------------------------------------------------------------------
    # Build dataset_statistics.json
    # -----------------------------------------------------------------------
    stats = {
        "total_images": total_images,
        "num_breeds": len(breeds),
        "breed_distribution": dict(breed_counts),
        "dataset_contribution_overall": dict(dataset_counts_overall),
        "dataset_contribution_per_breed": {
            b: dict(breed_dataset_matrix[b]) for b in breeds
        },
        "class_imbalance": {
            "max_breed": max_breed, "max_count": breed_counts[max_breed],
            "min_breed": min_breed, "min_count": breed_counts[min_breed],
            "imbalance_ratio": round(imbalance_ratio, 2),
            "mean_count_per_breed": round(mean_count, 1),
        },
        "source_concentration_flags": source_concentration_flags,
        "resolution_breakdown_per_breed": {
            b: dict(breed_resolution[b]) for b in breeds
        },
        "resolution_skew_flags": resolution_skew_flags,
        "grayscale_pct_per_breed": breed_grayscale_pct,
        "grayscale_overall_pct": round(overall_grayscale_pct, 3),
        "grayscale_skew_flags": grayscale_flags,
        "real_vs_synthetic_per_breed": real_synth_ratio,
    }

    with open(output_dir / "dataset_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # -----------------------------------------------------------------------
    # Charts
    # -----------------------------------------------------------------------
    breeds_sorted = sorted(breeds, key=lambda b: -breed_counts[b])

    # 1. Breed distribution
    plt.figure(figsize=(10, 6))
    plt.bar(breeds_sorted, [breed_counts[b] for b in breeds_sorted], color="#4C72B0")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Image count")
    plt.title("Breed Distribution")
    plt.tight_layout()
    plt.savefig(charts_dir / "breed_distribution.png", dpi=150)
    plt.close()

    # 2. Dataset contribution stacked per breed
    all_datasets = sorted(dataset_counts_overall.keys())
    plt.figure(figsize=(12, 7))
    bottom = [0] * len(breeds_sorted)
    for ds in all_datasets:
        values = [breed_dataset_matrix[b].get(ds, 0) for b in breeds_sorted]
        plt.bar(breeds_sorted, values, bottom=bottom, label=ds)
        bottom = [b + v for b, v in zip(bottom, values)]
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Image count")
    plt.title("Dataset Contribution per Breed")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(charts_dir / "dataset_contribution_per_breed.png", dpi=150)
    plt.close()

    # 3. Real vs synthetic composition
    plt.figure(figsize=(10, 6))
    real_vals = [real_synth_ratio[b]["real"] for b in breeds_sorted]
    synth_vals = [real_synth_ratio[b]["synthetic"] for b in breeds_sorted]
    plt.bar(breeds_sorted, real_vals, label="real (unique+canonical)", color="#55A868")
    plt.bar(breeds_sorted, synth_vals, bottom=real_vals, label="synthetic (augmented+topup)", color="#C44E52")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Image count")
    plt.title("Real vs Synthetic Composition per Breed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts_dir / "source_type_composition.png", dpi=150)
    plt.close()

    # 4. Resolution breakdown per breed
    plt.figure(figsize=(10, 6))
    res_categories = ["small", "medium", "large"]
    colors = ["#DD8452", "#8172B2", "#4C72B0"]
    bottom = [0] * len(breeds_sorted)
    for cat, color in zip(res_categories, colors):
        values = [breed_resolution[b].get(cat, 0) for b in breeds_sorted]
        plt.bar(breeds_sorted, values, bottom=bottom, label=cat, color=color)
        bottom = [b + v for b, v in zip(bottom, values)]
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Image count")
    plt.title("Resolution Category Breakdown per Breed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts_dir / "resolution_breakdown.png", dpi=150)
    plt.close()

    # 5. Grayscale % by breed
    plt.figure(figsize=(10, 6))
    gvals = [breed_grayscale_pct[b] * 100 for b in breeds_sorted]
    plt.bar(breeds_sorted, gvals, color="#8C8C8C")
    plt.axhline(overall_grayscale_pct * 100, color="red", linestyle="--", label="dataset average")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Grayscale %")
    plt.title("Grayscale Image % per Breed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts_dir / "grayscale_by_breed.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------------------
    # dataset_health_report.md
    # -----------------------------------------------------------------------
    lines = []
    lines.append("# Dataset Health Report\n")
    lines.append(f"**Total images:** {total_images}  \n**Breeds:** {len(breeds)}\n")

    lines.append("\n## Class Imbalance")
    lines.append(f"- Largest breed: **{max_breed}** ({breed_counts[max_breed]} images)")
    lines.append(f"- Smallest breed: **{min_breed}** ({breed_counts[min_breed]} images)")
    lines.append(f"- Imbalance ratio: **{imbalance_ratio:.2f}x**")
    lines.append(f"- Mean per breed: {mean_count:.1f}")

    lines.append("\n## Source-Concentration Risk")
    if source_concentration_flags:
        lines.append(f"⚠ {len(source_concentration_flags)} breed(s) rely on a single source dataset "
                      f"for ≥{int(SOURCE_CONCENTRATION_THRESHOLD*100)}% of their images:\n")
        for b, info in source_concentration_flags.items():
            lines.append(f"- **{b}**: {info['share']*100:.0f}% from `{info['dominant_dataset']}`")
    else:
        lines.append("✅ No breed is overly dependent on a single source dataset.")

    lines.append("\n## Resolution Consistency")
    if resolution_skew_flags:
        lines.append(f"⚠ {len(resolution_skew_flags)} breed(s) skew toward smaller images vs. dataset average "
                      f"({overall_small_pct*100:.1f}%):\n")
        for b, info in resolution_skew_flags.items():
            lines.append(f"- **{b}**: {info['small_pct']*100:.1f}% small images")
    else:
        lines.append("✅ No breed shows a significant resolution skew.")

    lines.append("\n## Grayscale Concentration")
    lines.append(f"Overall grayscale share: **{overall_grayscale_pct*100:.2f}%**")
    if grayscale_flags:
        lines.append(f"\n⚠ {len(grayscale_flags)} breed(s) have disproportionately high grayscale share:\n")
        for b, info in grayscale_flags.items():
            lines.append(f"- **{b}**: {info['grayscale_pct']*100:.1f}% (vs {info['dataset_avg_grayscale_pct']*100:.1f}% average)")
    else:
        lines.append("✅ Grayscale images are evenly distributed, not concentrated in any one breed.")

    lines.append("\n## Real vs Synthetic Composition (relevant for 07_split_dataset.py)")
    lines.append("| Breed | Real | Synthetic | Real % |")
    lines.append("|---|---|---|---|")
    for b in breeds_sorted:
        info = real_synth_ratio[b]
        lines.append(f"| {b} | {info['real']} | {info['synthetic']} | {info['real_pct']*100:.1f}% |")

    with open(output_dir / "dataset_health_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("DATASET STATISTICS SUMMARY")
    print("=" * 60)
    print(f"Total images: {total_images}  |  Breeds: {len(breeds)}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}x ({max_breed} vs {min_breed})")
    print(f"Source-concentration flags: {len(source_concentration_flags)}")
    print(f"Resolution-skew flags: {len(resolution_skew_flags)}")
    print(f"Grayscale-skew flags: {len(grayscale_flags)}  (overall grayscale: {overall_grayscale_pct*100:.2f}%)")
    print(f"\ndataset_statistics.json -> {output_dir / 'dataset_statistics.json'}")
    print(f"dataset_health_report.md -> {output_dir / 'dataset_health_report.md'}")
    print(f"Charts -> {charts_dir}")


if __name__ == "__main__":
    main()
