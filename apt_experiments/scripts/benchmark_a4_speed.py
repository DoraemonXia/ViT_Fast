"""Benchmark full ViT-B/16 and trained A4 Learned APT on the same GPU.

This script performs inference only. It does not train, update, or save model
weights, and it never needs to download pretrained backbone weights.
"""

import argparse
import os
import statistics
import sys
import time
from contextlib import nullcontext

import torch
import timm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APT_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(APT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from apt_experiments.Hierarchical_16_32_Learned_APT.train import (  # noqa: E402
    DATASETS,
    HierarchicalAPTViT,
    get_normalization,
    write_result_json,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure Baseline ViT-B/16 and A4 Learned APT speed under the "
            "same inference configuration. No training is performed."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS),
    )
    parser.add_argument(
        "--apt_checkpoint",
        required=True,
        help="A4 checkpoint_epoch_*.pth or best_model.pth",
    )
    parser.add_argument(
        "--baseline_checkpoint",
        default=None,
        help=(
            "Optional fine-tuned full ViT checkpoint. It is needed only when "
            "--measure_accuracy is used; speed can be measured without it."
        ),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--threshold32", type=float, default=None)
    parser.add_argument("--entropy_bins", type=int, default=None)
    parser.add_argument(
        "--backbone",
        default="vit_base_patch16_224.augreg_in21k",
    )
    parser.add_argument(
        "--warmup_batches",
        type=int,
        default=20,
        help="Warm-up batches before each timed repetition",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=100,
        help="Timed batches per repetition; 0 means the complete test set",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed repetitions; the median is reported",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable BF16 autocast",
    )
    parser.add_argument(
        "--measure_accuracy",
        action="store_true",
        help="Also evaluate test accuracy in an untimed pass",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSON path. Default: "
            "apt_experiments/benchmark_results/<dataset>_a4_speed.json"
        ),
    )
    return parser.parse_args()


def load_checkpoint(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary")
    for key in ("model_state_dict", "state_dict", "model", "model_ema"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    return {
        key.removeprefix("module.").removeprefix("model."): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }


def checkpoint_args(checkpoint):
    args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    return args if isinstance(args, dict) else {}


def load_model_state(model, checkpoint, strict, label):
    state_dict = extract_state_dict(checkpoint)
    if strict:
        model.load_state_dict(state_dict, strict=True)
        print(f"[{label}] Loaded {len(state_dict)} tensors (strict=True)")
        return

    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    if not compatible:
        raise RuntimeError(f"No compatible tensors found for {label}")
    incompatible = model.load_state_dict(compatible, strict=False)
    print(
        f"[{label}] Loaded {len(compatible)}/{len(model_state)} compatible "
        "tensors"
    )
    if incompatible.missing_keys:
        print(f"[{label}] Missing tensors: {len(incompatible.missing_keys)}")


def build_test_loader(args):
    loader_fn, _, _ = DATASETS[args.dataset]
    kwargs = {
        "batch_size": args.batch_size,
        "data_dir": os.path.join(PROJECT_ROOT, "data"),
        "num_workers": args.num_workers,
        "image_size": args.image_size,
    }
    if args.dataset == "cifar100":
        kwargs["return_val"] = True
    result = loader_fn(**kwargs)
    if len(result) == 4:
        _, _, test_loader, num_classes = result
    else:
        _, test_loader, num_classes = result
    return test_loader, num_classes


def autocast_context(use_amp):
    if not use_amp:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


@torch.inference_mode()
def warm_up(model, loader, device, use_amp, batches):
    model.eval()
    if batches <= 0:
        return
    completed = 0
    while completed < batches:
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            with autocast_context(use_amp):
                model(images)
            completed += 1
            if completed >= batches:
                break
    torch.cuda.synchronize(device)


@torch.inference_mode()
def timed_pass(model, loader, device, use_amp, max_batches, track_tokens):
    model.eval()
    start_events = []
    end_events = []
    batch_sizes = []
    real_token_sum = 0.0
    padded_token_sum = 0.0
    token_samples = 0

    torch.cuda.reset_peak_memory_stats(device)
    for batch_index, (images, _) in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with autocast_context(use_amp):
            model(images)
        end.record()

        start_events.append(start)
        end_events.append(end)
        batch_sizes.append(images.size(0))

        if track_tokens:
            counts = model._last_token_counts
            real_token_sum += counts.detach().float().sum().item()
            padded_token_sum += float(model._last_k * images.size(0))
            token_samples += images.size(0)

    if not batch_sizes:
        raise RuntimeError("No benchmark batches were produced")
    torch.cuda.synchronize(device)

    elapsed_ms = sum(
        start.elapsed_time(end)
        for start, end in zip(start_events, end_events)
    )
    samples = sum(batch_sizes)
    result = {
        "samples": samples,
        "batches": len(batch_sizes),
        "total_time_ms": elapsed_ms,
        "latency_ms_per_sample": elapsed_ms / samples,
        "throughput_samples_per_second": samples * 1000.0 / elapsed_ms,
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
    }
    if track_tokens:
        result.update({
            "avg_real_tokens": real_token_sum / token_samples,
            "avg_padded_tokens": padded_token_sum / token_samples,
            "original_tokens": int(model.num_patches),
        })
    return result


def median_result(results):
    keys = (
        "total_time_ms",
        "latency_ms_per_sample",
        "throughput_samples_per_second",
        "peak_memory_mb",
    )
    output = dict(results[0])
    for key in keys:
        output[key] = statistics.median(item[key] for item in results)
    output["all_repetitions"] = results
    return output


def benchmark(model, loader, device, use_amp, args, track_tokens=False):
    repetitions = []
    for repeat in range(args.repeats):
        warm_up(
            model,
            loader,
            device,
            use_amp,
            args.warmup_batches,
        )
        result = timed_pass(
            model,
            loader,
            device,
            use_amp,
            args.max_batches,
            track_tokens,
        )
        repetitions.append(result)
        print(
            f"  repeat {repeat + 1}/{args.repeats}: "
            f"{result['throughput_samples_per_second']:.1f} samples/s, "
            f"{result['latency_ms_per_sample']:.3f} ms/sample"
        )
    return median_result(repetitions)


@torch.inference_mode()
def evaluate_accuracy(model, loader, device, use_amp):
    model.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(use_amp):
            logits = model(images)
        correct += logits.argmax(dim=1).eq(targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct / total


def move_model_off_gpu(model):
    model.to("cpu")
    torch.cuda.empty_cache()
    time.sleep(1)


def print_summary(payload):
    baseline = payload["baseline"]
    apt = payload["a4_learned_apt"]
    print("\n| Method | Tokens (real/padded) | Accuracy | Latency | Throughput | Speedup |")
    print("|---|---:|---:|---:|---:|---:|")
    baseline_acc = (
        f"{baseline['test_accuracy']:.2f}%"
        if baseline.get("test_accuracy") is not None else "-"
    )
    apt_acc = (
        f"{apt['test_accuracy']:.2f}%"
        if apt.get("test_accuracy") is not None else "-"
    )
    print(
        "| Baseline ViT-B/16 | 196/196 | "
        f"{baseline_acc} | {baseline['latency_ms_per_sample']:.3f} ms | "
        f"{baseline['throughput_samples_per_second']:.1f}/s | 1.00x |"
    )
    print(
        "| A4 Learned APT | "
        f"{apt['avg_real_tokens']:.1f}/{apt['avg_padded_tokens']:.1f} | "
        f"{apt_acc} | {apt['latency_ms_per_sample']:.3f} ms | "
        f"{apt['throughput_samples_per_second']:.1f}/s | "
        f"{payload['speedup_ratio']:.3f}x "
        f"({payload['speedup_percent']:+.1f}%) |"
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this benchmark")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.warmup_batches < 0 or args.max_batches < 0:
        raise ValueError("Batch counts cannot be negative")

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_amp = not args.no_amp

    apt_checkpoint = load_checkpoint(args.apt_checkpoint)
    saved_args = checkpoint_args(apt_checkpoint)
    threshold32 = (
        args.threshold32
        if args.threshold32 is not None
        else float(saved_args.get("threshold32", 3.25))
    )
    entropy_bins = (
        args.entropy_bins
        if args.entropy_bins is not None
        else int(saved_args.get("entropy_bins", 64))
    )
    saved_dataset = saved_args.get("dataset")
    if saved_dataset and saved_dataset != args.dataset:
        raise ValueError(
            f"Checkpoint dataset is {saved_dataset}, not {args.dataset}"
        )

    loader, num_classes = build_test_loader(args)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(
        f"Dataset: {args.dataset}, batch={args.batch_size}, "
        f"AMP={'BF16' if use_amp else 'FP32'}, timed batches="
        f"{args.max_batches or 'all'}"
    )

    print("\n[1/2] Baseline ViT-B/16")
    baseline = timm.create_model(
        args.backbone,
        pretrained=False,
        num_classes=num_classes,
        img_size=args.image_size,
    ).to(device)
    baseline_has_weights = args.baseline_checkpoint is not None
    if baseline_has_weights:
        baseline_checkpoint = load_checkpoint(args.baseline_checkpoint)
        load_model_state(
            baseline,
            baseline_checkpoint,
            strict=False,
            label="BASELINE",
        )
        del baseline_checkpoint
    baseline_metrics = benchmark(
        baseline,
        loader,
        device,
        use_amp,
        args,
    )
    baseline_metrics["checkpoint"] = args.baseline_checkpoint
    baseline_metrics["test_accuracy"] = None
    if args.measure_accuracy:
        if not baseline_has_weights:
            print(
                "[BASELINE] Accuracy skipped because --baseline_checkpoint "
                "was not supplied"
            )
        else:
            baseline_metrics["test_accuracy"] = evaluate_accuracy(
                baseline, loader, device, use_amp
            )
    move_model_off_gpu(baseline)
    del baseline

    print("\n[2/2] A4 Hierarchical 16/32 Learned APT")
    apt = HierarchicalAPTViT(
        num_classes=num_classes,
        thresholds={32: threshold32},
        patch_sizes=(16, 32),
        aggregation="learned",
        use_scale_encoding=True,
        img_size=args.image_size,
        pretrained=False,
        input_mean=get_normalization(args.dataset)[0],
        input_std=get_normalization(args.dataset)[1],
        entropy_bins=entropy_bins,
        backbone_name=args.backbone,
        pretrained_checkpoint=None,
    ).to(device)
    load_model_state(apt, apt_checkpoint, strict=True, label="A4")
    del apt_checkpoint
    apt_metrics = benchmark(
        apt,
        loader,
        device,
        use_amp,
        args,
        track_tokens=True,
    )
    apt_metrics["checkpoint"] = args.apt_checkpoint
    apt_metrics["threshold32"] = threshold32
    apt_metrics["entropy_bins"] = entropy_bins
    apt_metrics["test_accuracy"] = None
    if args.measure_accuracy:
        apt_metrics["test_accuracy"] = evaluate_accuracy(
            apt, loader, device, use_amp
        )

    speedup_ratio = (
        apt_metrics["throughput_samples_per_second"]
        / baseline_metrics["throughput_samples_per_second"]
    )
    payload = {
        "dataset": args.dataset,
        "gpu": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "precision": "bf16" if use_amp else "fp32",
        "warmup_batches": args.warmup_batches,
        "timed_batches": args.max_batches or "all",
        "repeats": args.repeats,
        "baseline": baseline_metrics,
        "a4_learned_apt": apt_metrics,
        "speedup_ratio": speedup_ratio,
        "speedup_percent": (speedup_ratio - 1.0) * 100.0,
        "latency_reduction_percent": (
            1.0
            - apt_metrics["latency_ms_per_sample"]
            / baseline_metrics["latency_ms_per_sample"]
        )
        * 100.0,
    }

    output = args.output or os.path.join(
        APT_ROOT,
        "benchmark_results",
        f"{args.dataset}_a4_speed.json",
    )
    write_result_json(output, payload)
    print_summary(payload)
    print(f"\nSaved: {os.path.abspath(output)}")


if __name__ == "__main__":
    main()
