"""Aggregate APT results into machine-readable files and Markdown tables."""

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
    "threshold",
    "aggregation",
    "best_epoch",
    "best_val_acc",
    "val_acc_diff",
    "test_acc",
    "baseline_acc",
    "acc_diff",
    "original_tokens",
    "avg_real_tokens",
    "avg_padded_tokens",
    "token_ratio",
    "token_reduction_pct",
    "token_p50",
    "token_p90",
    "latency_ms",
    "throughput",
    "peak_memory_mb",
    "source",
]


def normalize_result(payload, path):
    args = payload.get("args") or {}
    token_stats = payload.get("token_stats") or {}
    original_tokens = payload.get("original_tokens")
    avg_real_tokens = payload.get("avg_real_tokens")
    token_ratio = payload.get("token_ratio")
    if token_ratio is None and original_tokens and avg_real_tokens is not None:
        token_ratio = avg_real_tokens / original_tokens
    return {
        "method": payload.get("method"),
        "dataset": payload.get("dataset"),
        "seed": payload.get("seed"),
        "threshold": payload.get(
            "threshold", args.get("threshold", args.get("threshold32"))
        ),
        "aggregation": args.get("aggregation"),
        "best_epoch": payload.get("best_epoch"),
        "best_val_acc": payload.get("best_val_acc"),
        "val_acc_diff": payload.get("val_acc_diff"),
        "test_acc": payload.get("test_acc"),
        "baseline_acc": payload.get("baseline_acc"),
        "acc_diff": payload.get("acc_diff"),
        "original_tokens": original_tokens,
        "avg_real_tokens": avg_real_tokens,
        "avg_padded_tokens": payload.get("avg_padded_tokens"),
        "token_ratio": token_ratio,
        "token_reduction_pct": (
            (1.0 - token_ratio) * 100.0 if token_ratio is not None else None
        ),
        "token_p50": token_stats.get("p50"),
        "token_p90": token_stats.get("p90"),
        "latency_ms": payload.get("latency_ms"),
        "throughput": payload.get("throughput"),
        "peak_memory_mb": payload.get("peak_memory_mb"),
        "source": os.path.relpath(path, APT_ROOT).replace(os.sep, "/"),
    }


def display(value, decimals=2, suffix=""):
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    return f"{float(value):.{decimals}f}{suffix}"


def method_name(row):
    names = {
        "a3_average_apt": "A3 Hierarchical Average",
        "a4_learned_apt": "A4 Hierarchical Learned",
        "historical_merge": "Historical Fixed Merge",
    }
    name = names.get(row["method"], row["method"] or "-")
    return name


def token_text(row):
    tokens = row.get("avg_real_tokens")
    ratio = row.get("token_ratio")
    if tokens is None:
        return "-"
    if ratio is None:
        return display(tokens, 1)
    return f"{float(tokens):.1f} ({float(ratio) * 100:.1f}%)"


def write_markdown(path, rows):
    lines = [
        "# APT 实验结果汇总",
        "",
        "## 精度与 Token",
        "",
        "| 数据集 | 方法 | Threshold | 原始 tokens | 处理后 tokens | Best Val Acc | vs Baseline | Test Acc |",
        "|:-------|:-----|----------:|------------:|--------------:|-------------:|------------:|---------:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {method} | {threshold} | {original} | {tokens} | "
            "{val} | {val_diff} | {test} |".format(
                dataset=row.get("dataset") or "-",
                method=method_name(row),
                threshold=display(row.get("threshold"), 2),
                original=display(row.get("original_tokens"), 0),
                tokens=token_text(row),
                val=display(row.get("best_val_acc"), 2, "%"),
                val_diff=display(row.get("val_acc_diff"), 2, "%"),
                test=display(row.get("test_acc"), 2, "%"),
            )
        )

    lines.extend([
        "",
        "## 资源消耗",
        "",
        "| 数据集 | 方法 | Real/Padded Tokens | 延迟 ms/sample | 吞吐 samples/s | 峰值显存 MB |",
        "|:-------|:-----|-------------------:|---------------:|---------------:|------------:|",
    ])
    for row in rows:
        real_padded = (
            f"{display(row.get('avg_real_tokens'), 1)} / "
            f"{display(row.get('avg_padded_tokens'), 1)}"
        )
        lines.append(
            "| {dataset} | {method} | {tokens} | {latency} | {throughput} | "
            "{memory} |".format(
                dataset=row.get("dataset") or "-",
                method=method_name(row),
                tokens=real_padded,
                latency=display(row.get("latency_ms"), 3),
                throughput=display(row.get("throughput"), 2),
                memory=display(row.get("peak_memory_mb"), 1),
            )
        )

    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=APT_ROOT)
    parser.add_argument(
        "--output_dir", default=os.path.join(APT_ROOT, "experiments", "results")
    )
    parser.add_argument(
        "--references",
        default=os.path.join(APT_ROOT, "experiments", "references"),
    )
    args = parser.parse_args()

    output_json = os.path.join(args.output_dir, "results.json")
    paths = sorted(glob.glob(
        os.path.join(args.root, "**", "results.json"), recursive=True
    ))
    paths = [
        path for path in paths
        if os.path.abspath(path) != os.path.abspath(output_json)
    ]
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as file:
            rows.append(normalize_result(json.load(file), path))
    for path in sorted(glob.glob(os.path.join(args.references, "*.json"))):
        with open(path, encoding="utf-8") as file:
            rows.append(normalize_result(json.load(file), path))

    rows.sort(key=lambda row: (
        row.get("dataset") or "",
        row.get("method") or "",
        row.get("seed") or 0,
    ))
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "results.csv")
    json_path = output_json
    markdown_path = os.path.join(args.output_dir, "RESULTS_TABLE.md")

    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)
        file.write("\n")
    write_markdown(markdown_path, rows)

    print(f"Aggregated {len(rows)} result files")
    print(csv_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
