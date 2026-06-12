"""Aggregate experiment results.json files into CSV and JSON."""

import argparse
import csv
import glob
import json
import os

APT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


FIELDS = [
    "method",
    "dataset",
    "seed",
    "best_val_acc",
    "test_acc",
    "baseline_acc",
    "acc_diff",
    "latency_ms",
    "throughput",
    "source",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.path.join(APT_ROOT, "checkpoints"))
    parser.add_argument(
        "--output_dir", default=os.path.join(APT_ROOT, "experiments", "results")
    )
    args = parser.parse_args()

    rows = []
    for path in glob.glob(
        os.path.join(args.root, "**", "results.json"), recursive=True
    ):
        with open(path, encoding="utf-8") as file:
            payload = json.load(file)
        row = {field: payload.get(field) for field in FIELDS}
        row["source"] = path
        rows.append(row)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "results.csv")
    json_path = os.path.join(args.output_dir, "results.json")
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(f"Aggregated {len(rows)} result files")
    print(csv_path)


if __name__ == "__main__":
    main()
