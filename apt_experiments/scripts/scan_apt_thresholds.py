"""Scan entropy thresholds and map them to token budgets without training."""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

APT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(APT_ROOT)
sys.path.insert(0, APT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from Hierarchical_16_32_Learned_APT.train import (
    compute_patch_entropy,
    denormalize_to_255,
    get_normalization,
)
from datasets import (
    get_cifar100_loader,
    get_dtd_loader,
    get_food101_loader,
    get_oxford_pets_loader,
)


DATASETS = {
    "cifar100": get_cifar100_loader,
    "oxford_pets": get_oxford_pets_loader,
    "food101": get_food101_loader,
    "dtd": get_dtd_loader,
}
TARGET_RATIOS = (0.75, 0.60, 0.50, 0.375)


def parse_thresholds(spec):
    start, stop, step = (float(value) for value in spec.split(":"))
    return np.arange(start, stop + step / 2, step).round(6).tolist()


def describe(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
        "min": int(array.min()),
        "max": int(array.max()),
    }


def a4_token_counts(entropy32, threshold, base_tokens):
    merged = (entropy32.flatten(1) < threshold).sum(dim=1)
    return base_tokens - 3 * merged


def find_candidates(rows, method, base_tokens):
    method_rows = [row for row in rows if row["method"] == method]
    candidates = []
    for ratio in TARGET_RATIOS:
        target = base_tokens * ratio
        best = min(method_rows, key=lambda row: abs(row["mean_tokens"] - target))
        candidates.append({
            "target_ratio": ratio,
            "target_tokens": target,
            "threshold": best["threshold"],
            "mean_tokens": best["mean_tokens"],
            "actual_ratio": best["mean_tokens"] / base_tokens,
        })
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--data_dir", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--thresholds", default="0.0:6.0:0.25")
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument(
        "--output_dir", default=os.path.join(APT_ROOT, "experiments", "scans")
    )
    args = parser.parse_args()

    loader_fn = DATASETS[args.dataset]
    kwargs = dict(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    if args.dataset == "cifar100":
        kwargs["return_val"] = True
    loaders = loader_fn(**kwargs)
    if len(loaders) != 4:
        raise RuntimeError("threshold scans require an independent validation set")
    _, val_loader, _, _ = loaders

    mean, std = get_normalization(args.dataset)
    entropy16_batches = []
    entropy32_batches = []
    seen = 0
    for images, _ in val_loader:
        remaining = args.max_samples - seen
        if remaining <= 0:
            break
        images = images[:remaining]
        pixels = denormalize_to_255(images, mean, std)
        maps = compute_patch_entropy(
            pixels, patch_sizes=(16, 32), bins=args.bins
        )
        entropy16_batches.append(maps[16].cpu())
        entropy32_batches.append(maps[32].cpu())
        seen += images.shape[0]

    entropy16 = torch.cat(entropy16_batches)
    entropy32 = torch.cat(entropy32_batches)
    base_tokens = entropy16.shape[1] * entropy16.shape[2]
    thresholds = parse_thresholds(args.thresholds)
    rows = []
    for threshold in thresholds:
        counts = a4_token_counts(entropy32, threshold, base_tokens)
        stats = describe(counts.numpy())
        rows.append({
            "dataset": args.dataset,
            "method": "a4_learned_apt",
            "threshold": threshold,
            "samples": seen,
            "base_tokens": base_tokens,
            "mean_tokens": stats["mean"],
            "token_ratio": stats["mean"] / base_tokens,
            "std_tokens": stats["std"],
            "p50_tokens": stats["p50"],
            "p90_tokens": stats["p90"],
            "min_tokens": stats["min"],
            "max_tokens": stats["max"],
        })

    payload = {
        "dataset": args.dataset,
        "image_size": args.image_size,
        "samples": seen,
        "bins": args.bins,
        "normalization": {"mean": mean, "std": std},
        "threshold_spec": args.thresholds,
        "entropy": {
            "patch16": describe(entropy16.numpy().reshape(-1)),
            "patch32": describe(entropy32.numpy().reshape(-1)),
        },
        "candidates": {
            "a4_learned_apt": find_candidates(
                rows, "a4_learned_apt", base_tokens
            )
        },
        "rows": rows,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    stem = f"{args.dataset}_img{args.image_size}_bins{args.bins}"
    json_path = os.path.join(args.output_dir, f"{stem}.json")
    csv_path = os.path.join(args.output_dir, f"{stem}.csv")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Scanned {seen} validation images from {args.dataset}")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    for method, candidates in payload["candidates"].items():
        print(method)
        for candidate in candidates:
            print(
                f"  target={candidate['target_ratio']:.3f} "
                f"threshold={candidate['threshold']:.2f} "
                f"tokens={candidate['mean_tokens']:.1f} "
                f"ratio={candidate['actual_ratio']:.3f}"
            )


if __name__ == "__main__":
    main()
