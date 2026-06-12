"""Generate resumable GPU experiment queues from CPU threshold scans."""

import argparse
import glob
import json
import os

APT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(APT_ROOT)


DATASET_EPOCHS = {
    "cifar100": 100,
    "oxford_pets": 100,
    "food101": 30,
    "dtd": 100,
}


def load_scans(scan_dir):
    scans = {}
    for path in glob.glob(os.path.join(scan_dir, "*.json")):
        with open(path, encoding="utf-8") as file:
            payload = json.load(file)
        scans[payload["dataset"]] = payload
    return scans


def candidate_map(scan, method):
    return {
        str(candidate["target_ratio"]): candidate
        for candidate in scan["candidates"][method]
    }


def command_for(entry):
    common = (
        f"--dataset {entry['dataset']} --gpu 0 "
        f"--batch_size {entry['batch_size']} --accum {entry['accum']} "
        f"--epochs {entry['epochs']} --seed {entry['seed']} "
        f"--entropy_bins {entry['entropy_bins']}"
    )
    if entry["method"] == "selection":
        return (
            f"python apt_experiments/train_apt_patch_selection.py {common} "
            f"--threshold {entry['threshold']}"
        )
    if entry["method"] == "merge":
        return (
            f"python apt_experiments/train_apt_patch_merge.py {common} "
            f"--threshold {entry['threshold']}"
        )
    return (
        f"python apt_experiments/train_hierarchical_apt.py {common} "
        f"--threshold32 {entry['threshold']} "
        f"--aggregation {entry['aggregation']}"
    )


def project_relative(path):
    return os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan_dir", default=os.path.join(APT_ROOT, "experiments", "scans")
    )
    parser.add_argument(
        "--output_dir", default=os.path.join(APT_ROOT, "experiments", "queues")
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["cifar100", "oxford_pets"]
    )
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.75])
    parser.add_argument("--short_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args()

    scans = load_scans(args.scan_dir)
    queue = []
    missing = []
    for dataset in args.datasets:
        if dataset not in scans:
            missing.append(dataset)
            continue
        scan = scans[dataset]
        for method in ("selection", "merge"):
            candidates = candidate_map(scan, method)
            for ratio in args.ratios:
                key = str(ratio)
                if key not in candidates:
                    continue
                candidate = candidates[key]
                for seed in args.seeds:
                    base = {
                        "dataset": dataset,
                        "method": method,
                        "target_ratio": ratio,
                        "threshold": candidate["threshold"],
                        "expected_ratio": candidate["actual_ratio"],
                        "entropy_bins": scan["bins"],
                        "batch_size": args.batch_size,
                        "accum": args.accum,
                        "seed": seed,
                        "epochs": args.short_epochs,
                        "stage": "short_screen",
                    }
                    queue.append(base)
                    if method == "merge":
                        hierarchical = dict(base)
                        hierarchical.update({
                            "method": "hierarchical",
                            "aggregation": "average",
                        })
                        queue.append(hierarchical)

    for entry in queue:
        entry["command"] = command_for(entry)

    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, "gpu_short_screen.jsonl")
    ps1_path = os.path.join(args.output_dir, "gpu_short_screen.ps1")
    sh_path = os.path.join(args.output_dir, "gpu_short_screen.sh")
    manifest_path = os.path.join(args.output_dir, "manifest.json")

    with open(jsonl_path, "w", encoding="utf-8") as file:
        for entry in queue:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open(ps1_path, "w", encoding="utf-8") as file:
        file.write("$ErrorActionPreference = 'Stop'\n")
        for entry in queue:
            file.write(entry["command"] + "\n")
    with open(sh_path, "w", encoding="utf-8", newline="\n") as file:
        file.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        for entry in queue:
            file.write(entry["command"] + "\n")
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump({
            "queue_size": len(queue),
            "datasets_ready": sorted(scans),
            "datasets_missing_scans": missing,
            "short_epochs": args.short_epochs,
            "files": [
                project_relative(jsonl_path),
                project_relative(ps1_path),
                project_relative(sh_path),
            ],
        }, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"Generated {len(queue)} GPU commands")
    print(f"Missing scans: {missing or 'none'}")
    print(project_relative(manifest_path))


if __name__ == "__main__":
    main()
