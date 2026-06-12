"""Shared utilities for APT-style adaptive patch tokenization."""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F


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
    """Compute grayscale Shannon entropy for non-overlapping patch grids.

    The scatter-add implementation avoids allocating a large one-hot tensor.
    """
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
    """Build a boolean self-attention mask that excludes padded keys.

    `valid_tokens` has shape (B, N). The returned shape (B, 1, 1, N) is
    broadcast across attention heads and query positions.
    """
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
        real = real_counts.detach().to("cpu", torch.float32).reshape(-1)
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
    import json
    import os

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
