# ---------------------------------------------------------------------------
# Example run:
# python 08_generate_clients.py
#
# Requires: pip install numpy matplotlib
# ---------------------------------------------------------------------------

"""
08_generate_clients.py

Purpose (per roadmap step 9 — Federated Client Generator)
---------------------------------------------------------------
Partitions master_dataset/train/ (ONLY train — val/test stay centralized,
shared across every client and any centralized baseline run) into N
simulated federated clients ("farms"). Every client gets all 15 breeds,
just in different (skewed) proportions — never a one-breed-only client.

Design (agreed through discussion, extends the base roadmap ask)
--------------------------------------------------------------------
1. Dirichlet-distribution skewing (standard FL-literature method, not
   hand-picked numbers): each breed's per-client proportions are sampled
   from Dirichlet(alpha). Low alpha = harsh Non-IID skew, high alpha =
   near-IID. Reproducible via RANDOM_SEED.
2. Guaranteed minimum per breed per client (MIN_CLIENT_SHARE): prevents
   Dirichlet sampling from accidentally starving a client down to ~0 of
   some breed, which would violate "every client contains all breeds."
3. Cluster-aware assignment: a duplicate/augmented-sibling cluster is
   assigned to ONE client as a whole (more realistic — photos of the same
   individual animal belong to one farm, not scattered across farms).
4. COPIES files (does not move) — train/ remains fully intact for
   centralized-vs-federated comparison (per your paper's Table 3/4 design).
5. Supports both 'iid' and 'non_iid' modes per the roadmap's explicit ask.

Output
------
- master_dataset/clients/client_<i>/train/<breed>/ for i in 1..NUM_CLIENTS
- client_distribution.csv: client_id, breed, count
- client_summary.json: full stats, config used
- client_distribution_chart.png: stacked bar, breed composition per client

Usage
-----
1. Edit SPLIT_MANIFEST_CSV, MASTER_DATASET_ROOT below.
2. Set NUM_CLIENTS, MODE ('iid' or 'non_iid'), DIRICHLET_ALPHA.
3. Run: python 08_generate_clients.py
"""

import csv
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("Missing dependencies. Install with:\n    pip install numpy matplotlib")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

SPLIT_MANIFEST_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\split_manifest.csv"
MASTER_DATASET_ROOT = r"D:\Projects\Federated_Learning\Master-Dataset\master_dataset"
CLIENTS_ROOT = r"D:\Projects\Federated_Learning\Master-Dataset\master_dataset\clients"

CLIENT_DISTRIBUTION_CSV = r"D:\Projects\Federated_Learning\Master-Dataset\master_dataset\clients\client_distribution.csv"
CLIENT_SUMMARY_JSON = r"D:\Projects\Federated_Learning\Master-Dataset\master_dataset\clients\client_summary.json"
CLIENT_CHART_PNG = r"D:\Projects\Federated_Learning\Master-Dataset\master_dataset\clients\client_distribution_chart.png"

NUM_CLIENTS = 10
MODE = "non_iid"          # 'iid' or 'non_iid'
DIRICHLET_ALPHA = 0.5     # lower = more skewed. ~0.1 harsh, ~1.0 moderate, 10+ ~IID
MIN_CLIENT_SHARE = 0.02   # each client guaranteed at least this fraction of each breed's total
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Cluster-aware, target-proportion assignment (reused pattern from 07)
# ---------------------------------------------------------------------------

def assign_clusters_to_clients(clusters, client_targets, rng):
    """
    clusters: dict cluster_key -> list of rows
    client_targets: dict client_id -> target image count (float)
    Returns dict cluster_key -> client_id
    Greedy: smallest clusters first, each assigned fully to whichever
    client has the largest remaining deficit as a fraction of its target.
    """
    remaining = dict(client_targets)
    cluster_items = list(clusters.items())
    rng.shuffle(cluster_items)
    cluster_items.sort(key=lambda kv: len(kv[1]))

    assignment = {}
    for cluster_key, rows in cluster_items:
        size = len(rows)
        best_client = max(
            remaining,
            key=lambda c: remaining[c] / client_targets[c] if client_targets[c] > 0 else -999
        )
        assignment[cluster_key] = best_client
        remaining[best_client] -= size

    return assignment


def compute_targets_iid(total, num_clients):
    base = total / num_clients
    return {i: base for i in range(1, num_clients + 1)}


def compute_targets_non_iid(total, num_clients, alpha, min_share, np_rng):
    proportions = np_rng.dirichlet([alpha] * num_clients)
    # enforce floor, then renormalize the remainder proportionally
    floor = min_share
    proportions = np.maximum(proportions, floor)
    proportions = proportions / proportions.sum()
    targets = proportions * total
    return {i + 1: targets[i] for i in range(num_clients)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = random.Random(RANDOM_SEED)
    np_rng = np.random.default_rng(RANDOM_SEED)

    with open(SPLIT_MANIFEST_CSV, newline="", encoding="utf-8") as f:
        train_rows = [r for r in csv.DictReader(f) if r["split"] == "train"]

    print(f"Loaded {len(train_rows)} TRAIN images (val/test untouched, stay centralized).")
    print(f"Mode: {MODE}  |  Clients: {NUM_CLIENTS}"
          + (f"  |  Dirichlet alpha: {DIRICHLET_ALPHA}" if MODE == "non_iid" else ""))

    # group by breed -> cluster -> rows
    breed_clusters = defaultdict(lambda: defaultdict(list))
    for row in train_rows:
        breed_clusters[row["breed"]][row["cluster_key"]].append(row)

    client_assignment = {}   # path -> client_id
    for breed, clusters in breed_clusters.items():
        total = sum(len(v) for v in clusters.values())
        if MODE == "iid":
            targets = compute_targets_iid(total, NUM_CLIENTS)
        else:
            targets = compute_targets_non_iid(total, NUM_CLIENTS, DIRICHLET_ALPHA, MIN_CLIENT_SHARE, np_rng)

        assignment = assign_clusters_to_clients(clusters, targets, rng)
        for cluster_key, client_id in assignment.items():
            for row in clusters[cluster_key]:
                client_assignment[row["path"]] = client_id

    # --- Copy files into clients/client_<i>/train/<breed>/ ---
    print("\nCopying files into client folders (train/ remains untouched)...")
    copied = 0
    distribution_rows = []
    breed_client_counts = defaultdict(lambda: defaultdict(int))

    for row in train_rows:
        client_id = client_assignment[row["path"]]
        src = Path(row["path"])
        dest_dir = Path(CLIENTS_ROOT) / f"client_{client_id}" / "train" / row["breed"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name

        if src.exists():
            try:
                shutil.copy2(str(src), str(dest))
                copied += 1
                breed_client_counts[row["breed"]][client_id] += 1
            except Exception as e:
                print(f"  [WARNING] failed to copy {src}: {e}")
        else:
            print(f"  [WARNING] source missing: {src}")

    for breed, client_counts in breed_client_counts.items():
        for client_id, count in client_counts.items():
            distribution_rows.append({"client_id": client_id, "breed": breed, "count": count})

    with open(CLIENT_DISTRIBUTION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["client_id", "breed", "count"])
        writer.writeheader()
        writer.writerows(distribution_rows)

    # --- Summary JSON ---
    breeds_sorted = sorted(breed_client_counts.keys())
    client_totals = defaultdict(int)
    for breed, cc in breed_client_counts.items():
        for cid, cnt in cc.items():
            client_totals[cid] += cnt

    summary = {
        "config": {
            "num_clients": NUM_CLIENTS, "mode": MODE,
            "dirichlet_alpha": DIRICHLET_ALPHA if MODE == "non_iid" else None,
            "min_client_share": MIN_CLIENT_SHARE, "random_seed": RANDOM_SEED,
        },
        "total_images_distributed": copied,
        "client_totals": dict(client_totals),
        "breed_distribution_per_client": {
            b: dict(breed_client_counts[b]) for b in breeds_sorted
        },
    }
    with open(CLIENT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # --- Chart ---
    client_ids = list(range(1, NUM_CLIENTS + 1))
    plt.figure(figsize=(12, 7))
    bottom = [0] * NUM_CLIENTS
    for breed in breeds_sorted:
        values = [breed_client_counts[breed].get(cid, 0) for cid in client_ids]
        plt.bar([f"client_{c}" for c in client_ids], values, bottom=bottom, label=breed)
        bottom = [b + v for b, v in zip(bottom, values)]
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Image count")
    plt.title(f"Breed Composition per Client ({MODE}, {NUM_CLIENTS} clients)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(CLIENT_CHART_PNG, dpi=150)
    plt.close()

    # --- Console summary ---
    print("\n" + "=" * 80)
    print("CLIENT GENERATION SUMMARY")
    print("=" * 80)
    header = "breed".ljust(22) + "".join(f"c{c}".rjust(7) for c in client_ids)
    print(header)
    for breed in breeds_sorted:
        line = breed.ljust(22) + "".join(str(breed_client_counts[breed].get(c, 0)).rjust(7) for c in client_ids)
        print(line)
    print("-" * len(header))
    totals_line = "TOTAL".ljust(22) + "".join(str(client_totals.get(c, 0)).rjust(7) for c in client_ids)
    print(totals_line)

    print(f"\nTotal images copied: {copied}")
    print(f"Clients folder: {CLIENTS_ROOT}")
    print(f"Distribution CSV: {CLIENT_DISTRIBUTION_CSV}")
    print(f"Summary JSON: {CLIENT_SUMMARY_JSON}")
    print(f"Chart: {CLIENT_CHART_PNG}")


if __name__ == "__main__":
    main()
