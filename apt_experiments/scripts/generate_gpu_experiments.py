"""Generate the A4-only GPU experiment queue."""

import argparse
import glob
import json
import os

APT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(APT_ROOT)


def load_scans(scan_dir):
    scans = {}
    for path in glob.glob(os.path.join(scan_dir, "*.json")):
        with open(path, encoding="utf-8-sig") as file:
            payload = json.load(file)
        scans[payload["dataset"]] = payload
    return scans


def threshold_candidate(scan, ratio):
    candidates = scan["candidates"]["a4_learned_apt"]
    return min(
        candidates,
        key=lambda candidate: abs(float(candidate["target_ratio"]) - ratio),
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--pretrained_checkpoint",
        default=None,
        help="Local pretrained ViT checkpoint on the GPU server",
    )
    args = parser.parse_args()

    scans = load_scans(args.scan_dir)
    queue = []
    missing = []
    for dataset in args.datasets:
        scan = scans.get(dataset)
        if scan is None:
            missing.append(dataset)
            continue
        for ratio in args.ratios:
            candidate = threshold_candidate(scan, ratio)
            for seed in args.seeds:
                entry = {
                    "dataset": dataset,
                    "method": "a4_learned_apt",
                    "target_ratio": ratio,
                    "threshold32": candidate["threshold"],
                    "expected_ratio": candidate["actual_ratio"],
                    "entropy_bins": scan["bins"],
                    "batch_size": args.batch_size,
                    "accum": args.accum,
                    "seed": seed,
                    "epochs": args.epochs,
                }
                entry["command"] = (
                    "python apt_experiments/Hierarchical_16_32_Learned_APT/train.py "
                    f"--dataset {dataset} --gpu 0 "
                    f"--batch_size {args.batch_size} --accum {args.accum} "
                    f"--epochs {args.epochs} --seed {seed} "
                    f"--entropy_bins {scan['bins']} "
                    f"--threshold32 {candidate['threshold']}"
                )
                if args.pretrained_checkpoint:
                    entry["pretrained_checkpoint"] = args.pretrained_checkpoint
                    entry["command"] += (
                        f" --pretrained_checkpoint "
                        f"{args.pretrained_checkpoint}"
                    )
                queue.append(entry)

    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, "gpu_a4.jsonl")
    ps1_path = os.path.join(args.output_dir, "gpu_a4.ps1")
    sh_path = os.path.join(args.output_dir, "gpu_a4.sh")
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
            "method": "a4_learned_apt",
            "datasets_missing_scans": missing,
            "epochs": args.epochs,
            "files": [
                project_relative(jsonl_path),
                project_relative(ps1_path),
                project_relative(sh_path),
            ],
        }, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"Generated {len(queue)} A4 GPU commands")
    print(f"Missing scans: {missing or 'none'}")
    print(project_relative(manifest_path))


if __name__ == "__main__":
    main()
