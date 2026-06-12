"""
Training-free token reduction for ViT-B/16.

This script evaluates plug-in token reduction methods without training:

1. ToMe-style per-layer token merging
   - merge r similar tokens inside every Transformer block
   - keep CLS token protected
   - track merged token size
   - use proportional attention: attn += log(token_size)

2. EViT-style CLS-attention pruning
   - run shallow blocks first
   - use CLS->patch attention at one block to keep high-attention patches
   - prune once, then run remaining blocks with fewer tokens

Keep these results separate from trained methods such as APT Entropy,
APT Merge, and MAE+Router unless the ToMe/EViT settings are fine-tuned too.
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import timm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import get_cifar100_loader, get_food101_loader, get_oxford_pets_loader

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - convenience fallback for minimal envs
    tqdm = None


HF_BASES = [
    "https://hf-mirror.com/1999xia/ViT_Fast/resolve/main",
    "https://huggingface.co/1999xia/ViT_Fast/resolve/main",
]

DATASETS = {
    "cifar100": (get_cifar100_loader, 100, "cifar100_vit_b16_ft"),
    "oxford_pets": (get_oxford_pets_loader, 37, "oxford_pets_vit_b16_ft"),
    "food101": (get_food101_loader, 101, "food101_vit_b16_in21k"),
}

CHECKPOINTS = {
    # CIFAR-100 is stored on Hugging Face as a single file without a .pth suffix.
    "cifar100_vit_b16_ft": {
        "local": [
            "checkpoints/cifar100_vit_b16_ft",
            "checkpoints/cifar100_vit_b16_ft/best_model.pth",
            "checkpoints/cifar100_vit_b16_ft.pth",
        ],
        "remote": [
            "checkpoints/cifar100_vit_b16_ft",
            "checkpoints/cifar100_vit_b16_ft/best_model.pth",
        ],
    },
    "oxford_pets_vit_b16_ft": {
        "local": [
            "checkpoints/oxford_pets_vit_b16_ft/best_model.pth",
            "checkpoints/oxford_pets_vit_b16_ft",
            "checkpoints/oxford_pets_vit_b16_ft.pth",
        ],
        "remote": [
            "checkpoints/oxford_pets_vit_b16_ft/best_model.pth",
            "checkpoints/oxford_pets_vit_b16_ft",
        ],
    },
    "food101_vit_b16_in21k": {
        "local": [
            "checkpoints/food101_vit_b16_in21k/best_model.pth",
            "checkpoints/food101_vit_b16_in21k",
            "checkpoints/food101_vit_b16_ft/best_model.pth",
            "checkpoints/food101_vit_b16_ft",
        ],
        "remote": [
            "checkpoints/food101_vit_b16_in21k/best_model.pth",
            "checkpoints/food101_vit_b16_in21k",
            "checkpoints/food101_vit_b16_ft/best_model.pth",
        ],
    },
}

TRAINED_REFERENCES = {
    "cifar100": [
        ("APT Entropy Selection", "trained 50 epochs", "133/196", "85.84%"),
        ("APT Merge", "trained 50 epochs", "73/196", "83.13%"),
        ("MAE+Router keep50", "trained", "98/196", "89.07%"),
        ("MAE+Router keep75", "trained", "147/196", "91.10%"),
    ],
}


def iter_loader(loader: Iterable, desc: str):
    if tqdm is None:
        return loader
    return tqdm(loader, total=len(loader), desc=desc, unit="batch", dynamic_ncols=True)


def parse_csv_floats(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_csv_ints(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def distribute_reductions(total_remove: int, depth: int) -> List[int]:
    """Spread token reductions across layers for a ToMe-style schedule."""
    if total_remove <= 0:
        return [0] * depth
    base = total_remove // depth
    rem = total_remove % depth
    return [base + (1 if i < rem else 0) for i in range(depth)]


def make_r_schedule(
    depth: int,
    r: Optional[int],
    target_keep_ratio: Optional[float],
    num_patches: int,
) -> List[int]:
    if r is not None:
        return [max(0, r)] * depth
    if target_keep_ratio is None:
        return [0] * depth
    target_keep = max(1, int(round(num_patches * target_keep_ratio)))
    total_remove = max(0, num_patches - target_keep)
    return distribute_reductions(total_remove, depth)


def checkpoint_spec(checkpoint_key: str) -> dict:
    if checkpoint_key in CHECKPOINTS:
        return CHECKPOINTS[checkpoint_key]
    return {
        "local": [
            f"checkpoints/{checkpoint_key}/best_model.pth",
            f"checkpoints/{checkpoint_key}",
            f"checkpoints/{checkpoint_key}.pth",
        ],
        "remote": [
            f"checkpoints/{checkpoint_key}/best_model.pth",
            f"checkpoints/{checkpoint_key}",
        ],
    }


def find_checkpoint(checkpoint_key: str) -> str:
    spec = checkpoint_spec(checkpoint_key)
    for path in spec["local"]:
        if os.path.isfile(path):
            return path
    return spec["local"][0]


def download_checkpoint(target_path: str, checkpoint_key: str) -> bool:
    if os.path.exists(target_path):
        return True
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    spec = checkpoint_spec(checkpoint_key)
    urls = [f"{base}/{path}" for base in HF_BASES for path in spec["remote"]]
    for url in urls:
        print(f"  Downloading: {url}", flush=True)
        try:
            urllib.request.urlretrieve(url, target_path)
            print(f"  Saved to: {target_path}", flush=True)
            return True
        except Exception as exc:
            print(f"  Failed: {exc}", flush=True)
    return False


def load_checkpoint_if_available(
    model: nn.Module,
    checkpoint: str,
    checkpoint_key: str,
    download: bool,
    timm_pretrained: bool,
) -> Optional[str]:
    if checkpoint == "none":
        if not timm_pretrained:
            print("  Warning: --checkpoint none and --timm_pretrained not set; evaluating random weights.", flush=True)
        return None
    ckpt_path = find_checkpoint(checkpoint_key) if checkpoint == "auto" else checkpoint
    if not os.path.exists(ckpt_path) and checkpoint == "auto" and download:
        download_checkpoint(ckpt_path, checkpoint_key)
    if not os.path.exists(ckpt_path):
        if timm_pretrained:
            print(
                f"  Warning: checkpoint not found ({ckpt_path}); using ImageNet-pretrained timm weights "
                "with a newly initialized dataset head.",
                flush=True,
            )
        else:
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. Pass --download, pass --checkpoint PATH, "
                "or pass --checkpoint none --timm_pretrained for an ImageNet-pretrained smoke test."
            )
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded checkpoint: {ckpt_path}", flush=True)
    if missing:
        print(f"  Missing keys: {len(missing)}", flush=True)
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}", flush=True)
    return ckpt_path


def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = True,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """ToMe bipartite matching. Protects CLS and optional distillation tokens."""
    protected = int(class_token) + int(distill_token)
    tokens = metric.shape[1]
    r = min(max(0, r), (tokens - protected) // 2)
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        a = metric[..., ::2, :]
        b = metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)
        if class_token:
            scores[..., 0, :] = -float("inf")
        if distill_token:
            scores[..., :, 0] = -float("inf")

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        batch, src_tokens, channels = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(batch, src_tokens - r, channels))
        src = src.gather(dim=-2, index=src_idx.expand(batch, r, channels))
        dst = dst.scatter_reduce(-2, dst_idx.expand(batch, r, channels), src, reduce=mode)
        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        batch, _, channels = unm.shape
        src = dst.gather(dim=-2, index=dst_idx.expand(batch, r, channels))
        out = torch.zeros(batch, metric.shape[1], channels, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(batch, unm_len, channels), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(batch, r, channels), src=src)
        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable,
    x: torch.Tensor,
    size: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge tokens using size-weighted averages and return updated sizes."""
    if size is None:
        size = torch.ones_like(x[..., :1])
    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size.clamp_min(1e-6)
    return x, size


class TrainingFreeReducerViT(nn.Module):
    """ViT-B/16 wrapper with training-free ToMe or EViT token reduction."""

    def __init__(
        self,
        num_classes: int,
        method: str,
        tome_r_schedule: Optional[Sequence[int]] = None,
        evit_keep_ratio: float = 0.75,
        evit_layer: int = 3,
        pretrained: bool = True,
    ):
        super().__init__()
        if method not in {"baseline", "tome", "evit"}:
            raise ValueError(f"Unknown method: {method}")
        self.method = method
        self.evit_keep_ratio = evit_keep_ratio
        self.evit_layer = evit_layer

        self.backbone = timm.create_model(
            "vit_base_patch16_224.augreg_in21k",
            pretrained=pretrained,
            num_classes=num_classes,
        )
        self.num_patches = self.backbone.patch_embed.num_patches
        depth = len(self.backbone.blocks)
        self.tome_r_schedule = list(tome_r_schedule or [0] * depth)
        if len(self.tome_r_schedule) != depth:
            raise ValueError(f"ToMe r schedule must have {depth} entries.")

        self._last_tokens = self.num_patches
        self._last_token_size_mean = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "baseline":
            self._last_tokens = self.num_patches
            return self.backbone(x)
        return self.forward_reduced(x)

    def forward_reduced(self, images: torch.Tensor) -> torch.Tensor:
        x = self.backbone.patch_embed(images)
        batch = x.shape[0]
        cls_token = self.backbone.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)

        size = None
        for idx, block in enumerate(self.backbone.blocks):
            if self.method == "tome":
                x, size = self.forward_tome_block(block, x, size, self.tome_r_schedule[idx])
            elif self.method == "evit":
                need_attn = idx == self.evit_layer
                x, cls_attn = self.forward_standard_block(block, x, need_attn=need_attn)
                if need_attn:
                    x = self.prune_by_cls_attention(x, cls_attn, self.evit_keep_ratio)
            else:
                raise ValueError(f"Unknown method: {self.method}")

        x = self.backbone.norm(x)
        x = self.backbone.head(x[:, 0])
        self._last_tokens = x.new_tensor(self._last_tokens).item()
        if size is not None:
            self._last_token_size_mean = size[:, 1:].float().mean().item()
        return x

    def forward_tome_block(
        self,
        block: nn.Module,
        x: torch.Tensor,
        size: Optional[torch.Tensor],
        r: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_size = size
        attn_out, metric, _ = self.attention_forward(block.attn, block.norm1(x), attn_size, False)
        x = x + self.drop_path1(block, attn_out)
        if r > 0:
            merge, _ = bipartite_soft_matching(metric, r, class_token=True, distill_token=False)
            x, size = merge_wavg(merge, x, size)
        x = x + self.drop_path2(block, block.mlp(block.norm2(x)))
        self._last_tokens = x.shape[1] - 1
        return x, size

    def forward_standard_block(
        self,
        block: nn.Module,
        x: torch.Tensor,
        need_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, _, cls_attn = self.attention_forward(block.attn, block.norm1(x), None, need_attn)
        x = x + self.drop_path1(block, attn_out)
        x = x + self.drop_path2(block, block.mlp(block.norm2(x)))
        return x, cls_attn

    @staticmethod
    def drop_path1(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if hasattr(block, "drop_path1"):
            return block.drop_path1(x)
        if hasattr(block, "drop_path"):
            return block.drop_path(x)
        return x

    @staticmethod
    def drop_path2(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if hasattr(block, "drop_path2"):
            return block.drop_path2(x)
        if hasattr(block, "drop_path"):
            return block.drop_path(x)
        return x

    @staticmethod
    def attention_forward(
        attn: nn.Module,
        x: torch.Tensor,
        size: Optional[torch.Tensor],
        need_attn: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch, tokens, channels = x.shape
        head_dim = channels // attn.num_heads
        qkv = attn.qkv(x).reshape(batch, tokens, 3, attn.num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q_norm = getattr(attn, "q_norm", None)
        k_norm = getattr(attn, "k_norm", None)
        if q_norm is not None:
            q = q_norm(q)
        if k_norm is not None:
            k = k_norm(k)

        scale = getattr(attn, "scale", head_dim ** -0.5)
        attn_logits = (q @ k.transpose(-2, -1)) * scale
        if size is not None:
            # Proportional attention: merged tokens represent multiple original tokens.
            attn_logits = attn_logits + size.log()[:, None, None, :, 0]

        attn_weights = attn_logits.softmax(dim=-1)
        cls_attn = attn_weights[:, :, 0, 1:].mean(dim=1) if need_attn else None
        attn_weights = attn.attn_drop(attn_weights)
        x = (attn_weights @ v).transpose(1, 2).reshape(batch, tokens, channels)
        x = attn.proj(x)
        x = attn.proj_drop(x)
        metric = k.mean(dim=1)
        return x, metric, cls_attn

    def prune_by_cls_attention(
        self,
        x: torch.Tensor,
        cls_attn: Optional[torch.Tensor],
        keep_ratio: float,
    ) -> torch.Tensor:
        if cls_attn is None:
            raise RuntimeError("EViT pruning requires CLS attention scores.")
        batch, tokens, channels = x.shape
        patch_tokens = tokens - 1
        keep = max(1, min(patch_tokens, int(round(patch_tokens * keep_ratio))))
        patch_idx = cls_attn.topk(keep, dim=1).indices + 1
        patch_idx = patch_idx.sort(dim=1)[0]
        cls_idx = torch.zeros(batch, 1, dtype=torch.long, device=x.device)
        gather_idx = torch.cat([cls_idx, patch_idx], dim=1)
        x = x.gather(dim=1, index=gather_idx.unsqueeze(-1).expand(-1, -1, channels))
        self._last_tokens = keep
        return x


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str, max_batches: Optional[int] = None):
    model.eval().to(device)
    correct = 0
    total = 0
    token_counts = []
    total_time = 0.0

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    batches = 0
    for batch_idx, (images, targets) in enumerate(iter_loader(loader, "Eval")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batches += 1
        images, targets = images.to(device), targets.to(device)
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        logits = model(images)
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time += time.time() - start

        pred = logits.argmax(dim=1)
        correct += pred.eq(targets).sum().item()
        total += targets.size(0)
        token_counts.append(float(getattr(model, "_last_tokens", 196)))

    acc = 100.0 * correct / max(1, total)
    avg_tokens = sum(token_counts) / max(1, len(token_counts))
    latency = total_time / max(1, total) * 1000.0
    throughput = total / total_time if total_time > 0 else 0.0
    return acc, avg_tokens, latency, throughput, total, batches


def get_test_loader(dataset: str, batch_size: int, num_workers: int):
    loader_fn, _, _ = DATASETS[dataset]
    result = loader_fn(batch_size=batch_size, data_dir="./data", num_workers=num_workers)
    if len(result) == 4:
        _, _, test_loader, num_classes = result
    else:
        _, test_loader, num_classes = result
    return test_loader, num_classes


def run_one(
    dataset: str,
    method: str,
    num_classes: int,
    device: str,
    test_loader,
    checkpoint: str,
    dataset_subpath: str,
    download: bool,
    max_batches: Optional[int],
    timm_pretrained: bool,
    tome_r: Optional[int] = None,
    tome_keep_ratio: Optional[float] = None,
    evit_keep_ratio: float = 0.75,
    evit_layer: int = 3,
):
    temp_model = timm.create_model("vit_base_patch16_224.augreg_in21k", pretrained=False, num_classes=num_classes)
    depth = len(temp_model.blocks)
    num_patches = temp_model.patch_embed.num_patches
    del temp_model
    schedule = make_r_schedule(depth, tome_r, tome_keep_ratio, num_patches)

    model = TrainingFreeReducerViT(
        num_classes=num_classes,
        method=method,
        tome_r_schedule=schedule,
        evit_keep_ratio=evit_keep_ratio,
        evit_layer=evit_layer,
        pretrained=timm_pretrained,
    )
    checkpoint_path = load_checkpoint_if_available(
        model.backbone, checkpoint, dataset_subpath, download, timm_pretrained)

    acc, avg_tokens, latency, throughput, samples, batches = evaluate(
        model, test_loader, device, max_batches)
    if method == "tome":
        setting = f"r={tome_r}" if tome_r is not None else f"target_keep={tome_keep_ratio:.2f}"
    elif method == "evit":
        setting = f"keep={evit_keep_ratio:.2f}, layer={evit_layer + 1}"
    else:
        setting = "full"
    print(
        f"{method:<9} {setting:<22} tokens={avg_tokens:6.1f}/{num_patches} "
        f"acc={acc:6.2f}% latency={latency:7.2f}ms throughput={throughput:7.1f}/s",
        flush=True,
    )
    result = {
        "dataset": dataset,
        "result_group": "training-free",
        "method": method,
        "setting": setting,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path or "",
        "timm_pretrained": timm_pretrained,
        "max_batches": max_batches,
        "samples": samples,
        "batches": batches,
        "tokens": round(avg_tokens, 4),
        "num_patches": num_patches,
        "token_keep_ratio": round(avg_tokens / num_patches, 6),
        "acc": round(acc, 4),
        "latency_ms": round(latency, 4),
        "throughput_s": round(throughput, 4),
        "device": device,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def save_results(results: List[dict], args: argparse.Namespace, results_dir: str) -> Tuple[str, str]:
    os.makedirs(results_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, "training_free_token_reduction.csv")
    jsonl_path = os.path.join(results_dir, "training_free_token_reduction.jsonl")
    rows = []
    for result in results:
        row = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "tome_r_values": args.tome_r_values,
            "tome_keep_ratios": args.tome_keep_ratios,
            "evit_keep_ratios": args.evit_keep_ratios,
            "evit_layer": args.evit_layer,
            **result,
        }
        rows.append(row)

    fieldnames = [
        "run_id", "timestamp", "dataset", "result_group", "method", "setting",
        "checkpoint", "checkpoint_path", "timm_pretrained", "batch_size", "num_workers",
        "max_batches", "samples", "batches", "tokens", "num_patches", "token_keep_ratio",
        "acc", "latency_ms", "throughput_s", "device", "tome_r_values", "tome_keep_ratios",
        "evit_keep_ratios", "evit_layer",
    ]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return csv_path, jsonl_path


def print_trained_references(dataset: str):
    refs = TRAINED_REFERENCES.get(dataset)
    if not refs:
        return
    print("\nTrained references (context only; do not directly rank against training-free results):")
    for name, training, tokens, acc in refs:
        print(f"  {name:<24} {training:<18} tokens={tokens:<8} acc={acc}")


def main():
    parser = argparse.ArgumentParser(description="Training-free ToMe/EViT token reduction evaluation")
    parser.add_argument("--dataset", type=str, default="cifar100", choices=list(DATASETS.keys()))
    parser.add_argument("--method", type=str, default="all", choices=["baseline", "tome", "evit", "all"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default="auto",
                        help="'auto' for checkpoints/{dataset}_vit_b16_ft, 'none', or a path")
    parser.add_argument("--download", dest="download", action="store_true", default=False,
                        help="Download missing baseline checkpoint from Hugging Face or hf-mirror")
    parser.add_argument("--no_download", dest="download", action="store_false",
                        help="Do not download missing checkpoints")
    parser.add_argument("--timm_pretrained", action="store_true",
                        help="Initialize timm pretrained weights before loading checkpoint")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Optional smoke-test limit for quick runs")
    parser.add_argument("--tome_r_values", type=str, default="0,4,8,13",
                        help="Comma-separated per-layer r values for ToMe")
    parser.add_argument("--tome_keep_ratios", type=str, default="",
                        help="Optional comma-separated target keep ratios converted to per-layer r schedules")
    parser.add_argument("--evit_keep_ratios", type=str, default="0.75,0.68,0.50",
                        help="Comma-separated patch keep ratios for EViT-style pruning")
    parser.add_argument("--evit_layer", type=int, default=3,
                        help="1-based block index whose CLS attention is used for pruning")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory for CSV/JSONL result records")
    parser.add_argument("--no_save_results", action="store_true",
                        help="Print results only; do not append CSV/JSONL records")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    print("Result group: training-free. Keep trained APT/MAE+Router in a separate table.", flush=True)

    test_loader, num_classes = get_test_loader(args.dataset, args.batch_size, args.num_workers)
    _, _, dataset_subpath = DATASETS[args.dataset]

    methods = ["baseline", "tome", "evit"] if args.method == "all" else [args.method]
    print("\nTraining-free results:")
    results = []

    if "baseline" in methods:
        results.append(run_one(
            args.dataset, "baseline", num_classes, device, test_loader,
            args.checkpoint, dataset_subpath, args.download, args.max_batches,
            args.timm_pretrained,
        ))

    if "tome" in methods:
        for r in parse_csv_ints(args.tome_r_values):
            results.append(run_one(
                args.dataset, "tome", num_classes, device, test_loader,
                args.checkpoint, dataset_subpath, args.download, args.max_batches,
                args.timm_pretrained,
                tome_r=r,
            ))
        for keep_ratio in parse_csv_floats(args.tome_keep_ratios):
            results.append(run_one(
                args.dataset, "tome", num_classes, device, test_loader,
                args.checkpoint, dataset_subpath, args.download, args.max_batches,
                args.timm_pretrained,
                tome_keep_ratio=keep_ratio,
            ))

    if "evit" in methods:
        prune_layer = max(0, args.evit_layer - 1)
        for keep_ratio in parse_csv_floats(args.evit_keep_ratios):
            results.append(run_one(
                args.dataset, "evit", num_classes, device, test_loader,
                args.checkpoint, dataset_subpath, args.download, args.max_batches,
                args.timm_pretrained,
                evit_keep_ratio=keep_ratio,
                evit_layer=prune_layer,
            ))

    if results and not args.no_save_results:
        csv_path, jsonl_path = save_results(results, args, args.results_dir)
        print(f"\nSaved result records:\n  CSV:   {csv_path}\n  JSONL: {jsonl_path}", flush=True)

    print_trained_references(args.dataset)


if __name__ == "__main__":
    main()
