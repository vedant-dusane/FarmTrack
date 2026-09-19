"""
reconcile.py

Purpose
-------
Compares the auto-generated dataset_inventory.csv (from 01_scan_datasets.py)
against the manually-counted Excel inventory, breed-by-breed, dataset-by-
dataset. Instead of trusting one total over the other, this pinpoints the
EXACT rows that disagree so you can go look at the actual folder and find
out why.

What it flags
--------------
1. Breeds where image_count differs between script and Excel.
2. Breed folders the script found that don't exist as a row in the Excel
   (e.g. stray folders, typos, case-mismatched duplicates).
3. Breeds listed in the Excel for a dataset but NOT found by the script
   (e.g. folder renamed differently than expected, or path wrong).

Usage
-----
"""
# python "D:\Projects\Federated_Learning\Master-Dataset\Scripts\01a_crosscheck_manual_script.py" --csv "D:\Projects\Federated_Learning\Master-Dataset\Scripts\Script_CountOf_Dataset.csv" --excel "D:\Projects\Federated_Learning\Unzip-Modified\Manual-CountOf-Datasets.xlsx"


import argparse
import csv
import sys
from collections import defaultdict

try:
    import openpyxl
except ImportError:
    sys.exit("This script needs openpyxl. Install with: pip install openpyxl")


def load_script_inventory(csv_path):
    """
    Returns dict: {(dataset, breed): image_count}
    """
    data = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row["dataset"].strip()
            breed = row["breed"].strip()
            count = int(row["image_count"])
            data[(dataset, breed)] = count
    return data


def load_excel_inventory(excel_path, sheet_name="Images Per Set"):
    """
    Reads the breed x dataset matrix sheet.
    Returns dict: {(dataset, breed): image_count}
    Row 2 (index) holds dataset names as column headers (starting col C / index 3).
    Breed names are in column B. '_' or blank means 0 / not present.
    """
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    ws = wb[sheet_name]

    rows = list(ws.iter_rows(values_only=True))

    # header row: find the row that has dataset names (skip title row)
    header_row = None
    for row in rows:
        if row[1] == "Names of Breed":
            header_row = row
            break
    if header_row is None:
        sys.exit("Could not find header row ('Names of Breed') in Excel sheet.")

    dataset_names = [c for c in header_row[2:] if c]

    data = {}
    header_seen = False
    for row in rows:
        if row[1] == "Names of Breed":
            header_seen = True
            continue
        if not header_seen:
            continue
        breed = row[1]
        if not breed:
            continue
        for i, dataset in enumerate(dataset_names):
            raw_val = row[2 + i]
            if raw_val in (None, "_", ""):
                count = 0
            else:
                try:
                    count = int(raw_val)
                except (ValueError, TypeError):
                    count = 0
            data[(dataset, breed)] = count

    return data


def reconcile(script_data, excel_data):
    all_keys = set(script_data) | set(excel_data)

    mismatches = []
    script_only = []
    excel_only = []

    for key in all_keys:
        dataset, breed = key
        script_count = script_data.get(key)
        excel_count = excel_data.get(key)

        if script_count is None:
            if excel_count:
                excel_only.append((dataset, breed, excel_count))
        elif excel_count is None:
            if script_count:
                script_only.append((dataset, breed, script_count))
        elif script_count != excel_count:
            mismatches.append((dataset, breed, script_count, excel_count,
                                script_count - excel_count))

    return mismatches, script_only, excel_only


def detect_swaps(mismatches, excel_data):
    swaps = []
    by_dataset_excel = defaultdict(lambda: defaultdict(list))
    for (dataset, breed), count in excel_data.items():
        if count:
            by_dataset_excel[dataset][count].append(breed)

    seen_pairs = set()
    for dataset, breed, script_count, excel_count, diff in mismatches:
        if script_count == 0:
            continue
        candidates = by_dataset_excel[dataset].get(script_count, [])
        for other_breed in candidates:
            if other_breed == breed:
                continue
            pair_key = tuple(sorted([breed, other_breed])) + (dataset,)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            swaps.append((dataset, breed, other_breed, script_count))

    return swaps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to Script_CountOf_Dataset.csv")
    parser.add_argument("--excel", required=True, help="Path to the Excel inventory")
    parser.add_argument("--sheet", default="Images Per Set",
                         help="Sheet name in Excel (default: 'Images Per Set')")
    args = parser.parse_args()

    script_data = load_script_inventory(args.csv)
    excel_data = load_excel_inventory(args.excel, args.sheet)

    mismatches, script_only, excel_only = reconcile(script_data, excel_data)
    swaps = detect_swaps(mismatches, excel_data)

    if swaps:
        print("=" * 80)
        print(f"⚠ SUSPECTED SWAPPED/MISLABELED FOLDERS — {len(swaps)} found")
        print("(one breed's script count exactly equals another breed's excel count")
        print(" in the same dataset — check these folders' actual images)")
        print("=" * 80)
        for dataset, breed, other_breed, count in swaps:
            print(f"[{dataset}] '{breed}' (script) <-> '{other_breed}' (excel)  both={count}")
        print()

    print("=" * 80)
    print(f"COUNT MISMATCHES (breed exists in both, counts differ) — {len(mismatches)} found")
    print("=" * 80)
    for dataset, breed, script_count, excel_count, diff in sorted(
            mismatches, key=lambda x: (x[0], -abs(x[4]))):
        sign = "+" if diff > 0 else ""
        print(f"[{dataset}] {breed:25s} script={script_count:5d}  excel={excel_count:5d}  diff={sign}{diff}")

    print()
    print("=" * 80)
    print(f"SCRIPT FOUND, MISSING FROM EXCEL (possible stray/typo folders) — {len(script_only)} found")
    print("=" * 80)
    for dataset, breed, count in sorted(script_only):
        print(f"[{dataset}] {breed:25s} count={count}")

    print()
    print("=" * 80)
    print(f"IN EXCEL, NOT FOUND BY SCRIPT (possible wrong path / renamed folder) — {len(excel_only)} found")
    print("=" * 80)
    for dataset, breed, count in sorted(excel_only):
        print(f"[{dataset}] {breed:25s} expected_count={count}")

    total_mismatch_images = sum(abs(d[4]) for d in mismatches)
    print()
    print(f"Total absolute image discrepancy across mismatched rows: {total_mismatch_images}")


if __name__ == "__main__":
    main()