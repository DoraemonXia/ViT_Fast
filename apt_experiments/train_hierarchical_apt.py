"""Train hierarchical APT with entropy-guided recursive patch subdivision.

CPU development should use ``--backbone vit_tiny_patch16_224 --no_pretrained``
only for smoke tests. Formal experiments use the default ViT-B/16 backbone on
GPU.
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import timm

APT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APT_ROOT)
import sys
sys.path.insert(0, APT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from apt_utils import (
    compute_patch_entropy,
    denormalize_to_255,
    get_normalization,
    run_masked_vit_blocks,
    write_result_json,
)
from train_apt_patch_merge import (
    BASELINE_ACC,
    DATASETS,
    compute_efficiency_metrics,
    evaluate,
    train_one_epoch,
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


class HierarchicalAPTViT(nn.Module):
    """ViT with entropy-guided hierarchical 16/32(/64) patch tokens."""

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
    ):
        super().__init__()
        self.img_size = img_size
        self.input_mean = input_mean
        self.input_std = input_std
        self.entropy_bins = entropy_bins
        self.aggregation = aggregation
        self.use_scale_encoding = use_scale_encoding

        backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            img_size=img_size,
        )
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

    def _region_indices(self, row, col, cells, device):
        grid = self.grid_size[1]
        rows = torch.arange(row, row + cells, device=device)
        cols = torch.arange(col, col + cells, device=device)
        return (rows[:, None] * grid + cols[None, :]).reshape(-1)

    def _build_image_tokens(self, image_index, fine_tokens, entropy_maps):
        tokens = []
        positions = []
        regions = []
        base_positions = self.pos_embed[0, 1:]
        largest_size = self.patch_sizes[-1]
        largest_cells = largest_size // self.base_patch_size

        def visit(row, col, patch_size):
            cells = patch_size // self.base_patch_size
            should_merge = False
            if patch_size > self.base_patch_size:
                entropy = entropy_maps[patch_size][
                    image_index, row // cells, col // cells
                ]
                should_merge = bool(entropy < self.thresholds[patch_size])

            if should_merge or patch_size == self.base_patch_size:
                indices = self._region_indices(
                    row, col, cells, fine_tokens.device
                )
                region_tokens = fine_tokens[image_index, indices]
                token = self._aggregate(
                    region_tokens.unsqueeze(0), patch_size
                ).squeeze(0)
                position = base_positions[indices].mean(dim=0)
                if self.use_scale_encoding:
                    position = position + self.scale_embed[
                        self.scale_to_index[patch_size]
                    ]
                tokens.append(token)
                positions.append(position)
                regions.append((row, col, cells, cells, patch_size))
                return

            child_size = patch_size // 2
            child_cells = cells // 2
            for row_offset in (0, child_cells):
                for col_offset in (0, child_cells):
                    visit(row + row_offset, col + col_offset, child_size)

        for row in range(0, self.grid_size[0], largest_cells):
            for col in range(0, self.grid_size[1], largest_cells):
                visit(row, col, largest_size)

        return torch.stack(tokens), torch.stack(positions), regions

    def forward(self, images):
        batch = images.shape[0]
        images_255 = denormalize_to_255(
            images, self.input_mean, self.input_std
        )
        entropy_maps = compute_patch_entropy(
            images_255,
            patch_sizes=self.patch_sizes,
            bins=self.entropy_bins,
        )
        fine_tokens = self.patch_embed(images)

        token_lists = []
        position_lists = []
        region_lists = []
        counts = []
        for image_index in range(batch):
            tokens, positions, regions = self._build_image_tokens(
                image_index, fine_tokens, entropy_maps
            )
            token_lists.append(tokens)
            position_lists.append(positions)
            region_lists.append(regions)
            counts.append(tokens.shape[0])

        max_tokens = max(counts)
        hidden = fine_tokens.new_zeros(batch, max_tokens, self.embed_dim)
        positions = fine_tokens.new_zeros(batch, max_tokens, self.embed_dim)
        valid = torch.zeros(
            batch, max_tokens, dtype=torch.bool, device=images.device
        )
        for image_index, count in enumerate(counts):
            hidden[image_index, :count] = token_lists[image_index]
            positions[image_index, :count] = position_lists[image_index]
            valid[image_index, :count] = True

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

        self._last_k = max_tokens
        self._last_n = self.num_patches
        self._last_token_counts = torch.tensor(
            counts, dtype=torch.long, device=images.device
        )
        self._last_regions = region_lists
        return logits


def parse_thresholds(args):
    thresholds = {32: args.threshold32}
    if args.image_size % 64 == 0 and args.enable_64:
        thresholds[64] = args.threshold64
    return thresholds


def main():
    parser = argparse.ArgumentParser(description="Hierarchical APT training")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--threshold32", type=float, default=5.0)
    parser.add_argument("--threshold64", type=float, default=5.0)
    parser.add_argument("--enable_64", action="store_true")
    parser.add_argument(
        "--aggregation", choices=["average", "learned"], default="average"
    )
    parser.add_argument("--no_scale_encoding", action="store_true")
    parser.add_argument("--entropy_bins", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accum", type=int, default=4)
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
    parser.add_argument("--no_pretrained", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available() and not args.eval_only:
        raise RuntimeError(
            "CUDA is required for training. CPU is limited to direct model smoke tests."
        )
    loader_fn, _, default_epochs = DATASETS[args.dataset]
    epochs = args.epochs or default_epochs
    loader_kwargs = dict(
        batch_size=args.batch_size,
        data_dir=os.path.join(PROJECT_ROOT, "data"),
        num_workers=4,
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

    patch_sizes = (16, 32, 64) if args.enable_64 else (16, 32)
    model = HierarchicalAPTViT(
        num_classes=num_classes,
        thresholds=parse_thresholds(args),
        patch_sizes=patch_sizes,
        aggregation=args.aggregation,
        use_scale_encoding=not args.no_scale_encoding,
        img_size=args.image_size,
        pretrained=not args.no_pretrained,
        input_mean=get_normalization(args.dataset)[0],
        input_std=get_normalization(args.dataset)[1],
        entropy_bins=args.entropy_bins,
        backbone_name=args.backbone,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    if args.eval_only:
        checkpoint = torch.load(args.eval_only, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, desc="Test"
        )
        print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.2f}%")
        return

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    save_dir = (
        os.path.join(
            APT_ROOT,
            "checkpoints",
            f"{args.dataset}_hierarchical_apt_{args.image_size}_"
            f"{args.aggregation}_t32_{args.threshold32}_s{args.seed}",
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
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
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
        )
        scheduler.step()
        val_loss, val_acc, avg_tokens, total_tokens, keep_pct, stats = evaluate(
            model, val_loader, criterion, device, track_patches=True, desc="Val"
        )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "token_stats": stats.to_dict(),
            "args": vars(args),
        }
        torch.save(
            checkpoint,
            os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth"),
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
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
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device, desc="Test"
    )
    latency, throughput = compute_efficiency_metrics(
        model, test_loader, device
    )
    baseline = BASELINE_ACC[args.dataset]
    write_result_json(os.path.join(save_dir, "results.json"), {
        "method": "hierarchical_apt",
        "dataset": args.dataset,
        "seed": args.seed,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_acc": test_acc,
        "baseline_acc": baseline,
        "acc_diff": test_acc - baseline,
        "latency_ms": latency,
        "throughput": throughput,
        "args": vars(args),
    })


if __name__ == "__main__":
    main()
