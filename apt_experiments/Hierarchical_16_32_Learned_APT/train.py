"""Train the primary A4 learned hierarchical APT experiment.

CPU development should use ``--backbone vit_tiny_patch16_224 --no_pretrained``
only for smoke tests. Formal experiments use the default ViT-B/16 backbone on
GPU.
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm import tqdm

EXPERIMENT_ROOT = os.path.dirname(os.path.abspath(__file__))
APT_ROOT = os.path.dirname(EXPERIMENT_ROOT)
PROJECT_ROOT = os.path.dirname(APT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from datasets import (
    get_cifar100_loader,
    get_dtd_loader,
    get_food101_loader,
    get_oxford_pets_loader,
)


DATASET_NORMALIZATION = {
    "cifar10": (
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010),
    ),
    "cifar100": (
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761),
    ),
    "imagenet": (
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),
}


DATASETS = {
    "cifar100": (get_cifar100_loader, 100, 100),
    "oxford_pets": (get_oxford_pets_loader, 37, 100),
    "food101": (get_food101_loader, 101, 30),
    "dtd": (get_dtd_loader, 47, 100),
}


BASELINE_ACC = {
    "cifar100": 91.69,
    "oxford_pets": 93.81,
    "food101": 91.37,
    "dtd": 80.85,
}


def get_normalization(dataset: str):
    """Return the normalization used by the project data loader."""
    if dataset in DATASET_NORMALIZATION:
        return DATASET_NORMALIZATION[dataset]
    return DATASET_NORMALIZATION["imagenet"]


def denormalize_to_255(
    images: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """Convert normalized images back to clamped RGB values in [0, 255]."""
    mean_tensor = images.new_tensor(mean).view(1, -1, 1, 1)
    std_tensor = images.new_tensor(std).view(1, -1, 1, 1)
    return ((images * std_tensor + mean_tensor) * 255.0).clamp_(0, 255)


def compute_patch_entropy(
    images_255: torch.Tensor,
    patch_sizes: Iterable[int] = (16,),
    bins: int = 64,
    pad_value: float = 1e6,
) -> Dict[int, torch.Tensor]:
    """Compute grayscale Shannon entropy for non-overlapping patch grids."""
    if images_255.ndim != 4:
        raise ValueError("images_255 must have shape (B, C, H, W)")
    if bins <= 1 or bins > 256:
        raise ValueError("bins must be in [2, 256]")

    batch, channels, height, width = images_255.shape
    if channels == 3:
        rgb_weights = images_255.new_tensor(
            [0.2989, 0.5870, 0.1140]
        ).view(1, 3, 1, 1)
        gray = (images_255 * rgb_weights).sum(dim=1)
    elif channels == 1:
        gray = images_255[:, 0]
    else:
        raise ValueError(f"expected 1 or 3 channels, got {channels}")

    entropy_maps = {}
    for patch_size in patch_sizes:
        if patch_size <= 0:
            raise ValueError("patch sizes must be positive")

        grid_h = (height + patch_size - 1) // patch_size
        grid_w = (width + patch_size - 1) // patch_size
        pad_h = grid_h * patch_size - height
        pad_w = grid_w * patch_size - width
        padded = F.pad(gray, (0, pad_w, 0, pad_h), value=0)

        patches = padded.unfold(1, patch_size, patch_size).unfold(
            2, patch_size, patch_size
        )
        pixels_per_patch = patch_size * patch_size
        quantized = (
            patches.reshape(batch, grid_h, grid_w, pixels_per_patch)
            .mul(bins / 256.0)
            .long()
            .clamp_(0, bins - 1)
        )

        num_blocks = batch * grid_h * grid_w
        flat = quantized.reshape(num_blocks, pixels_per_patch)
        offsets = (
            torch.arange(num_blocks, device=images_255.device).unsqueeze(1) * bins
        )
        histogram_indices = (flat + offsets).reshape(-1)
        histogram = torch.zeros(
            num_blocks * bins,
            device=images_255.device,
            dtype=torch.float32,
        )
        histogram.scatter_add_(
            0,
            histogram_indices,
            torch.ones_like(histogram_indices, dtype=torch.float32),
        )
        probabilities = histogram.reshape(
            batch, grid_h, grid_w, bins
        ) / pixels_per_patch
        entropy = -(probabilities * torch.log2(probabilities + 1e-10)).sum(dim=-1)

        if pad_h:
            entropy[:, -1, :] = pad_value
        if pad_w:
            entropy[:, :, -1] = pad_value
        entropy_maps[patch_size] = entropy

    return entropy_maps


def build_key_attention_mask(valid_tokens: torch.Tensor) -> torch.Tensor:
    """Build a boolean self-attention mask that excludes padded keys."""
    if valid_tokens.dtype != torch.bool or valid_tokens.ndim != 2:
        raise ValueError("valid_tokens must be a boolean tensor of shape (B, N)")
    return valid_tokens[:, None, None, :]


def run_masked_vit_blocks(blocks, hidden, valid_tokens):
    """Run timm ViT blocks while excluding padded keys and zeroing padded queries."""
    attention_mask = build_key_attention_mask(valid_tokens)
    query_mask = valid_tokens.unsqueeze(-1).to(hidden.dtype)
    for block in blocks:
        hidden = block(hidden, attn_mask=attention_mask)
        hidden = hidden * query_mask
    return hidden


@dataclass
class TokenStats:
    mean: float
    std: float
    p50: float
    p90: float
    minimum: int
    maximum: int
    padded_mean: float
    count: int

    def to_dict(self):
        return asdict(self)


class TokenStatsAccumulator:
    """Accumulate real and padded token counts across batches."""

    def __init__(self):
        self.real_counts = []
        self.padded_counts = []

    def update(self, real_counts: torch.Tensor, padded_length: Optional[int] = None):
        real = real_counts.detach().to(torch.float32).reshape(-1)
        self.real_counts.append(real)
        if padded_length is None:
            padded_length = int(real.max().item())
        self.padded_counts.append(
            torch.full_like(real, float(padded_length), dtype=torch.float32)
        )

    def compute(self) -> TokenStats:
        if not self.real_counts:
            raise RuntimeError("no token counts were recorded")
        real = torch.cat(self.real_counts)
        padded = torch.cat(self.padded_counts)
        return TokenStats(
            mean=float(real.mean().item()),
            std=float(real.std(unbiased=False).item()),
            p50=float(torch.quantile(real, 0.5).item()),
            p90=float(torch.quantile(real, 0.9).item()),
            minimum=int(real.min().item()),
            maximum=int(real.max().item()),
            padded_mean=float(padded.mean().item()),
            count=int(real.numel()),
        )


def write_result_json(path, payload):
    """Write a reproducible UTF-8 JSON result file."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")


def update_history_json(path, record):
    """Insert or replace one epoch record in a JSON training history."""
    history = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as file:
            history = json.load(file)
    history = [
        item for item in history if int(item.get("epoch", -1)) != int(record["epoch"])
    ]
    history.append(record)
    history.sort(key=lambda item: int(item["epoch"]))
    write_result_json(path, history)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    accum_steps=1,
    epoch=None,
    total_epochs=None,
    use_amp=True,
    log_interval=20,
):
    model.train()
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = 0
    optimizer.zero_grad(set_to_none=True)

    desc = f"Epoch {epoch}/{total_epochs}" if epoch is not None else "Train"
    pbar = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=desc,
        unit="batch",
        dynamic_ncols=True,
    )
    for batch_idx, (images, targets) in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            logits = model(images)
            full_loss = criterion(logits, targets)
            loss = full_loss / accum_steps
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += full_loss.detach()
        predicted = logits.argmax(dim=1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum()
        if (batch_idx + 1) % log_interval == 0:
            pbar.set_postfix({
                "loss": f"{total_loss.item() / (batch_idx + 1):.4f}",
                "acc": f"{100.0 * correct.item() / total:.1f}%",
                "tokens": f"{model._last_k}/{model._last_n}",
            })

    if (batch_idx + 1) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss.item() / len(loader), 100.0 * correct.item() / total


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    track_patches=False,
    desc="Eval",
    use_amp=True,
    log_interval=20,
):
    model.eval()
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = 0
    token_stats = TokenStatsAccumulator() if track_patches else None

    pbar = tqdm(loader, total=len(loader), desc=desc, unit="batch", dynamic_ncols=True)
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, targets)
        total_loss += loss
        predicted = logits.argmax(dim=1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum()
        if track_patches:
            token_stats.update(model._last_token_counts, model._last_k)
        if (batch_idx + 1) % log_interval == 0:
            pbar.set_postfix({
                "loss": f"{total_loss.item() / (batch_idx + 1):.4f}",
                "acc": f"{100.0 * correct.item() / total:.1f}%",
            })

    result = (
        total_loss.item() / len(loader),
        100.0 * correct.item() / total,
    )
    if track_patches:
        stats = token_stats.compute()
        result += (
            stats.mean,
            model._last_n,
            stats.mean / model._last_n * 100,
            stats,
        )
    return result


@torch.no_grad()
def compute_efficiency_metrics(model, loader, device, use_amp=True):
    model.eval()
    for images, _ in loader:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            _ = model(images.to(device, non_blocking=True))
        break
    if device != "cpu":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    total_time = 0.0
    total_samples = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        if device != "cpu":
            torch.cuda.synchronize()
        start = time.time()
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            _ = model(images)
        if device != "cpu":
            torch.cuda.synchronize()
        total_time += time.time() - start
        total_samples += images.size(0)

    return (
        total_time / total_samples * 1000,
        total_samples / total_time,
        (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if device != "cpu"
            else 0.0
        ),
    )


class LearnedTokenAggregator(nn.Module):
    """Pool a variable set of fine tokens with learned scalar attention."""

    def __init__(self, embed_dim):
        super().__init__()
        hidden_dim = max(32, embed_dim // 4)
        self.score = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens):
        weights = self.score(tokens).softmax(dim=1)
        return (tokens * weights).sum(dim=1)


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("pretrained checkpoint must contain a state dict")
    for key in (
        "model_state_dict",
        "state_dict",
        "model",
        "model_ema",
    ):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    return {
        key.removeprefix("module.").removeprefix("model."): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }


def load_local_pretrained(backbone, checkpoint_path):
    """Load matching ViT weights without requiring network access."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {checkpoint_path}"
        )

    if checkpoint_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise RuntimeError(
                "Loading .safetensors requires: pip install safetensors"
            ) from error
        state_dict = load_file(checkpoint_path, device="cpu")
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        state_dict = _extract_state_dict(checkpoint)

    model_state = backbone.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    if not compatible:
        raise RuntimeError(
            "No compatible ViT parameters were found in "
            f"{checkpoint_path}. Check the backbone architecture."
        )

    incompatible = backbone.load_state_dict(compatible, strict=False)
    print(
        f"[PRETRAINED] Loaded {len(compatible)}/{len(model_state)} tensors "
        f"from {checkpoint_path}",
        flush=True,
    )
    if incompatible.missing_keys:
        print(
            f"[PRETRAINED] Missing tensors: {len(incompatible.missing_keys)} "
            "(the classification head may differ, which is expected)",
            flush=True,
        )


class HierarchicalAPTViT(nn.Module):
    """ViT with entropy-guided hierarchical 16/32 patch tokens."""

    def __init__(
        self,
        num_classes=100,
        thresholds=None,
        patch_sizes=(16, 32),
        aggregation="average",
        use_scale_encoding=True,
        img_size=224,
        pretrained=True,
        drop_path_rate=0.0,
        input_mean=(0.485, 0.456, 0.406),
        input_std=(0.229, 0.224, 0.225),
        entropy_bins=64,
        backbone_name="vit_base_patch16_224.augreg_in21k",
        pretrained_checkpoint=None,
    ):
        super().__init__()
        self.img_size = img_size
        self.input_mean = input_mean
        self.input_std = input_std
        self.entropy_bins = entropy_bins
        self.aggregation = aggregation
        self.use_scale_encoding = use_scale_encoding

        create_kwargs = {
            "num_classes": num_classes,
            "drop_path_rate": drop_path_rate,
            "img_size": img_size,
        }
        if pretrained_checkpoint:
            backbone = timm.create_model(
                backbone_name,
                pretrained=False,
                **create_kwargs,
            )
            load_local_pretrained(backbone, pretrained_checkpoint)
        else:
            try:
                backbone = timm.create_model(
                    backbone_name,
                    pretrained=pretrained,
                    **create_kwargs,
                )
            except Exception as error:
                if pretrained:
                    raise RuntimeError(
                        "Unable to download the timm pretrained backbone. "
                        "The server cannot reach Hugging Face. Download the "
                        "ViT-B/16 IN-21K weights on a machine with network "
                        "access, upload them to the server, and rerun with "
                        "--pretrained_checkpoint /path/to/weights.pth. "
                        "Use --no_pretrained only for smoke tests."
                    ) from error
                raise
        self.patch_embed = backbone.patch_embed
        self.cls_token = backbone.cls_token
        self.pos_embed = backbone.pos_embed
        self.pos_drop = backbone.pos_drop
        self.blocks = backbone.blocks
        self.norm = backbone.norm
        self.head = backbone.head
        self.embed_dim = backbone.embed_dim
        self.grid_size = tuple(self.patch_embed.grid_size)
        self.num_patches = self.patch_embed.num_patches
        del backbone

        if self.grid_size[0] != self.grid_size[1]:
            raise ValueError("hierarchical APT currently requires a square grid")
        base_patch = self.patch_embed.patch_size
        if isinstance(base_patch, tuple):
            base_patch = base_patch[0]
        self.base_patch_size = int(base_patch)

        valid_sizes = sorted(set(int(size) for size in patch_sizes))
        if self.base_patch_size not in valid_sizes:
            valid_sizes.insert(0, self.base_patch_size)
        for size in valid_sizes:
            if size % self.base_patch_size:
                raise ValueError("patch sizes must be multiples of the base patch")
            cells = size // self.base_patch_size
            if cells & (cells - 1):
                raise ValueError("hierarchical patch ratios must be powers of two")
            if self.grid_size[0] % cells:
                raise ValueError(
                    f"{size}x{size} patches do not tile grid {self.grid_size}"
                )
        self.patch_sizes = tuple(valid_sizes)
        if self.patch_sizes != (
            self.base_patch_size,
            self.base_patch_size * 2,
        ):
            raise ValueError(
                "the optimized A4 path currently supports only 16/32 patches"
            )

        thresholds = thresholds or {}
        self.thresholds = {
            int(size): float(thresholds.get(int(size), 5.0))
            for size in self.patch_sizes
            if size > self.base_patch_size
        }

        num_scales = len(self.patch_sizes)
        self.scale_embed = nn.Parameter(
            torch.zeros(num_scales, self.embed_dim),
            requires_grad=use_scale_encoding,
        )
        self.scale_to_index = {
            size: index for index, size in enumerate(self.patch_sizes)
        }

        if aggregation == "average":
            self.aggregators = nn.ModuleDict()
        elif aggregation == "learned":
            self.aggregators = nn.ModuleDict({
                str(size): LearnedTokenAggregator(self.embed_dim)
                for size in self.patch_sizes
                if size > self.base_patch_size
            })
        else:
            raise ValueError("aggregation must be 'average' or 'learned'")

        self._last_k = self.num_patches
        self._last_n = self.num_patches
        self._last_token_counts = None
        self._last_regions = None

    def _aggregate(self, tokens, patch_size):
        if patch_size == self.base_patch_size or self.aggregation == "average":
            return tokens.mean(dim=1)
        return self.aggregators[str(patch_size)](tokens)

    def _build_compact_tokens(self, fine_tokens, merge_mask):
        """Build and compact all 16/32 candidates with batched tensor ops."""
        batch, _, embed_dim = fine_tokens.shape
        grid_h, grid_w = self.grid_size
        coarse_h, coarse_w = grid_h // 2, grid_w // 2

        region_tokens = (
            fine_tokens.reshape(batch, coarse_h, 2, coarse_w, 2, embed_dim)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, coarse_h * coarse_w, 4, embed_dim)
        )
        base_positions = (
            self.pos_embed[0, 1:]
            .reshape(coarse_h, 2, coarse_w, 2, embed_dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(coarse_h * coarse_w, 4, embed_dim)
        )

        coarse_size = self.base_patch_size * 2
        coarse_tokens = self._aggregate(
            region_tokens.reshape(-1, 4, embed_dim),
            coarse_size,
        ).reshape(batch, coarse_h * coarse_w, embed_dim)
        coarse_positions = base_positions.mean(dim=1)

        fine_positions = base_positions
        if self.use_scale_encoding:
            fine_positions = (
                fine_positions
                + self.scale_embed[self.scale_to_index[self.base_patch_size]]
            )
            coarse_positions = (
                coarse_positions
                + self.scale_embed[self.scale_to_index[coarse_size]]
            )

        first_tokens = torch.where(
            merge_mask.unsqueeze(-1),
            coarse_tokens,
            region_tokens[:, :, 0],
        )
        candidates = torch.cat(
            [first_tokens.unsqueeze(2), region_tokens[:, :, 1:]],
            dim=2,
        )
        first_positions = torch.where(
            merge_mask.unsqueeze(-1),
            coarse_positions.unsqueeze(0),
            fine_positions[:, 0].unsqueeze(0),
        )
        candidate_positions = torch.cat(
            [
                first_positions.unsqueeze(2),
                fine_positions[:, 1:].unsqueeze(0).expand(batch, -1, -1, -1),
            ],
            dim=2,
        )

        valid = torch.cat(
            [
                torch.ones_like(merge_mask).unsqueeze(-1),
                (~merge_mask).unsqueeze(-1).expand(-1, -1, 3),
            ],
            dim=2,
        )

        candidates = candidates.flatten(1, 2)
        candidate_positions = candidate_positions.flatten(1, 2)
        valid = valid.flatten(1)
        counts = valid.sum(dim=1)

        slot_count = valid.shape[1]
        slot_indices = torch.arange(slot_count, device=fine_tokens.device)
        sort_keys = slot_indices.unsqueeze(0) + (~valid) * slot_count
        order = sort_keys.argsort(dim=1)
        gather_index = order.unsqueeze(-1).expand(-1, -1, embed_dim)
        candidates = candidates.gather(1, gather_index)
        candidate_positions = candidate_positions.gather(1, gather_index)

        max_tokens = int(counts.max().item())
        candidates = candidates[:, :max_tokens]
        candidate_positions = candidate_positions[:, :max_tokens]
        compact_valid = (
            torch.arange(max_tokens, device=fine_tokens.device).unsqueeze(0)
            < counts.unsqueeze(1)
        )
        return candidates, candidate_positions, compact_valid, counts

    def forward(self, images):
        batch = images.shape[0]
        coarse_size = self.base_patch_size * 2
        images_255 = denormalize_to_255(
            images, self.input_mean, self.input_std
        )
        entropy_maps = compute_patch_entropy(
            images_255,
            patch_sizes=(coarse_size,),
            bins=self.entropy_bins,
        )
        fine_tokens = self.patch_embed(images)
        merge_mask = (
            entropy_maps[coarse_size] < self.thresholds[coarse_size]
        ).flatten(1)
        hidden, positions, valid, counts = self._build_compact_tokens(
            fine_tokens,
            merge_mask,
        )

        cls_token = self.cls_token.expand(batch, -1, -1)
        cls_position = self.pos_embed[:, :1].expand(batch, -1, -1)
        hidden = torch.cat([cls_token, hidden], dim=1)
        positions = torch.cat([cls_position, positions], dim=1)
        hidden = self.pos_drop(hidden + positions)
        sequence_valid = torch.cat([
            torch.ones(batch, 1, dtype=torch.bool, device=images.device),
            valid,
        ], dim=1)
        hidden = run_masked_vit_blocks(
            self.blocks, hidden, sequence_valid
        )
        logits = self.head(self.norm(hidden)[:, 0])

        self._last_k = hidden.shape[1] - 1
        self._last_n = self.num_patches
        self._last_token_counts = counts
        self._last_regions = None
        return logits


def parse_thresholds(args):
    return {32: args.threshold32}


def main(aggregation="learned"):
    parser = argparse.ArgumentParser(
        description=f"Hierarchical APT training ({aggregation} aggregation)"
    )
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--threshold32", type=float, default=5.0)
    parser.add_argument("--entropy_bins", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval_only", default=None)
    parser.add_argument(
        "--backbone", default="vit_base_patch16_224.augreg_in21k"
    )
    parser.add_argument(
        "--pretrained_checkpoint",
        default=None,
        help="Local ViT checkpoint; when provided, no online download is used",
    )
    parser.add_argument("--no_pretrained", action="store_true")
    args = parser.parse_args()
    args.aggregation = aggregation

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    use_amp = device != "cpu" and not args.no_amp
    if not torch.cuda.is_available() and not args.eval_only:
        raise RuntimeError(
            "CUDA is required for training. CPU is limited to direct model smoke tests."
        )
    loader_fn, _, default_epochs = DATASETS[args.dataset]
    epochs = args.epochs or default_epochs
    loader_kwargs = dict(
        batch_size=args.batch_size,
        data_dir=os.path.join(PROJECT_ROOT, "data"),
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    if args.dataset == "cifar100":
        loader_kwargs["return_val"] = True
    result = loader_fn(**loader_kwargs)
    if len(result) == 4:
        train_loader, val_loader, test_loader, num_classes = result
    else:
        train_loader, test_loader, num_classes = result
        raise RuntimeError("formal training requires an independent validation set")

    patch_sizes = (16, 32)
    model = HierarchicalAPTViT(
        num_classes=num_classes,
        thresholds=parse_thresholds(args),
        patch_sizes=patch_sizes,
        aggregation=aggregation,
        use_scale_encoding=True,
        img_size=args.image_size,
        pretrained=not args.no_pretrained,
        input_mean=get_normalization(args.dataset)[0],
        input_std=get_normalization(args.dataset)[1],
        entropy_bins=args.entropy_bins,
        backbone_name=args.backbone,
        pretrained_checkpoint=args.pretrained_checkpoint,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    if args.eval_only:
        checkpoint = torch.load(args.eval_only, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device,
            desc="Test",
            use_amp=use_amp,
            log_interval=args.log_interval,
        )
        print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.2f}%")
        return

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=device != "cpu",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    save_dir = (
        os.path.join(
            EXPERIMENT_ROOT,
            "checkpoints",
            f"{args.dataset}_{'a4_learned' if aggregation == 'learned' else 'a3_average'}_"
            f"{args.image_size}_t32_{args.threshold32}_s{args.seed}",
        )
    )
    os.makedirs(save_dir, exist_ok=True)
    write_result_json(os.path.join(save_dir, "args.json"), vars(args))

    start_epoch = 0
    best_val_acc = 0.0
    best_epoch = -1
    resume_path = args.resume
    if resume_path is None:
        candidates = sorted(
            (
                filename
                for filename in os.listdir(save_dir)
                if filename.startswith("checkpoint_epoch_")
                and filename.endswith(".pth")
            ),
            key=lambda filename: int(
                filename.removeprefix("checkpoint_epoch_").removesuffix(".pth")
            ),
        )
        if candidates:
            resume_path = os.path.join(save_dir, candidates[-1])
    if resume_path:
        print(f"[RESUME] Loading checkpoint: {resume_path}", flush=True)
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = max(
            checkpoint.get("best_val_acc", 0.0),
            checkpoint.get("val_acc", 0.0),
        )
        best_epoch = checkpoint.get("best_epoch", -1)

    for epoch in range(start_epoch, epochs):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.accum,
            epoch=epoch + 1,
            total_epochs=epochs,
            use_amp=use_amp,
            log_interval=args.log_interval,
        )
        scheduler.step()
        val_loss, val_acc, avg_tokens, total_tokens, keep_pct, stats = evaluate(
            model,
            val_loader,
            criterion,
            device,
            track_patches=True,
            desc="Val",
            use_amp=use_amp,
            log_interval=args.log_interval,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "token_stats": stats.to_dict(),
            "avg_tokens": avg_tokens,
            "avg_total": total_tokens,
            "keep_pct": keep_pct,
            "args": vars(args),
        }
        torch.save(
            checkpoint,
            os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth"),
        )
        update_history_json(os.path.join(save_dir, "history.json"), {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "original_tokens": int(total_tokens),
            "avg_real_tokens": stats.mean,
            "avg_padded_tokens": stats.padded_mean,
            "token_ratio": stats.mean / total_tokens,
            "token_stats": stats.to_dict(),
        })
        if val_acc == best_val_acc and best_epoch == epoch:
            checkpoint["test_acc"] = None
            torch.save(checkpoint, os.path.join(save_dir, "best_model.pth"))
        print(
            f"Epoch {epoch + 1}/{epochs} ({time.time() - start:.1f}s) "
            f"train={train_acc:.2f}% val={val_acc:.2f}% "
            f"tokens={avg_tokens:.1f}/{total_tokens} ({keep_pct:.1f}%)"
        )

    checkpoint = torch.load(
        os.path.join(save_dir, "best_model.pth"), map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    best_val_acc = checkpoint.get("val_acc", best_val_acc)
    best_epoch = checkpoint.get("epoch", best_epoch)
    best_token_stats = checkpoint.get("token_stats", {})
    original_tokens = int(checkpoint.get("avg_total", model.num_patches))
    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device,
        desc="Test",
        use_amp=use_amp,
        log_interval=args.log_interval,
    )
    latency, throughput, peak_memory_mb = compute_efficiency_metrics(
        model, test_loader, device, use_amp=use_amp
    )
    baseline = BASELINE_ACC[args.dataset]
    write_result_json(os.path.join(save_dir, "results.json"), {
        "method": "a4_learned_apt" if aggregation == "learned" else "a3_average_apt",
        "dataset": args.dataset,
        "seed": args.seed,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "val_acc_diff": best_val_acc - baseline,
        "test_acc": test_acc,
        "baseline_acc": baseline,
        "acc_diff": test_acc - baseline,
        "latency_ms": latency,
        "throughput": throughput,
        "peak_memory_mb": peak_memory_mb,
        "original_tokens": original_tokens,
        "avg_real_tokens": best_token_stats.get("mean"),
        "avg_padded_tokens": best_token_stats.get("padded_mean"),
        "token_ratio": (
            best_token_stats.get("mean") / original_tokens
            if best_token_stats.get("mean") is not None else None
        ),
        "token_stats": best_token_stats,
        "args": vars(args),
    })


if __name__ == "__main__":
    main(aggregation="learned")
