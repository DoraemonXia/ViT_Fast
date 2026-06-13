"""
Train ViT-B/16 with APT-style entropy-based multi-scale patch merge.

不同于 train_apt_patch_selection.py 的简单丢弃策略，本脚本实现了与原始 APT 论文
一致的多尺度合并策略：
  - 高熵 2x2 (16×16) patch 块 → 保持 4 个独立 16×16 tokens
  - 低熵 2x2 (16×16) patch 块 → 合并为 1 个 32×32 token（平均 pooling）
  - 有效减少 token 数量，同时保留低熵区域的结构信息

Architecture:
  Image → Patch Embed (ALL 196 patches, 16×16)
    → 熵值计算 (16×16 + 32×32 scale)
    → 按 32×32 熵值决定合并/保留:
        - 熵 < threshold → merge 4→1 token (avg pool)
        - 熵 >= threshold → keep 4 individual tokens
    → 混合序列: 16×16 tokens + merged tokens
    → Pos Embed (16×16 保留原 pos, merged 用 7×7 重采样 pos)
    → ViT Blocks → CLS → 分类

Usage:
  python train_apt_patch_merge.py --dataset cifar100 --gpu 0
  python train_apt_patch_merge.py --dataset oxford_pets --gpu 0 --threshold 5.0

  # 仅评估
  python train_apt_patch_merge.py --dataset cifar100 --gpu 0 --eval_only PATH
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import sys
import argparse
import json
import random

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import get_cifar100_loader, get_oxford_pets_loader, get_food101_loader
from tqdm import tqdm
import timm


NORMALIZATION = {
    'cifar100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    'oxford_pets': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'food101': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


# ==============================================================================
# APT Entropy Computation (shared with train_apt_patch_selection.py)
# ==============================================================================

def compute_patch_entropy_fast(images_255, patch_size=16, num_scales=2, bins=64):
    """Fast entropy via scatter_add (avoids huge one-hot tensor)."""
    B, C, H, W = images_255.shape
    device = images_255.device

    if C == 3:
        w_rgb = torch.tensor([0.2989, 0.5870, 0.1140], device=device).view(1, 3, 1, 1)
        gray = (images_255 * w_rgb).sum(dim=1)
    else:
        gray = images_255[:, 0]

    entropy_maps = {}
    patch_sizes = [patch_size * (2 ** i) for i in range(num_scales)]

    for ps in patch_sizes:
        n_h = (H + ps - 1) // ps
        n_w = (W + ps - 1) // ps
        pad_h = n_h * ps - H
        pad_w = n_w * ps - W
        padded = F.pad(gray, (0, pad_w, 0, pad_h), mode='constant', value=0)

        patches = padded.unfold(1, ps, ps).unfold(2, ps, ps)
        flat = patches.reshape(B, n_h, n_w, ps * ps)
        flat_int = (flat * (bins / 256.0)).long().clamp(0, bins - 1)

        N_blocks = B * n_h * n_w
        flat_2d = flat_int.reshape(N_blocks, ps * ps)  # (N, P)

        # Scatter-add histogram: offset each block's indices into flattened array
        offsets = torch.arange(N_blocks, device=device).unsqueeze(1) * bins
        idx = (flat_2d + offsets).reshape(-1)
        hist_flat = torch.zeros(N_blocks * bins, device=device, dtype=torch.float32)
        hist_flat.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
        hist = hist_flat.reshape(N_blocks, bins)

        hist = hist.reshape(B, n_h, n_w, bins)
        probs = hist / (ps * ps)
        eps = 1e-10
        emap = -torch.sum(probs * torch.log2(probs + eps), dim=3)

        if pad_h > 0:
            emap[:, -1, :] = 1e6
        if pad_w > 0:
            emap[:, :, -1] = 1e6

        entropy_maps[ps] = emap
    return entropy_maps


def load_local_pretrained(backbone, checkpoint_path):
    """Load matching timm ViT weights without network access."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f'Pretrained checkpoint not found: {checkpoint_path}')

    if checkpoint_path.endswith('.safetensors'):
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise RuntimeError(
                'Loading .safetensors requires: pip install safetensors'
            ) from error
        state_dict = load_file(checkpoint_path, device='cpu')
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint
        for key in ('model_state_dict', 'state_dict', 'model', 'model_ema'):
            if isinstance(state_dict, dict) and isinstance(
                    state_dict.get(key), dict):
                state_dict = state_dict[key]
                break

    if not isinstance(state_dict, dict):
        raise TypeError('pretrained checkpoint must contain a state dict')
    state_dict = {
        key.removeprefix('module.').removeprefix('model.'): value
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor)
    }
    model_state = backbone.state_dict()
    compatible = {
        key: value for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    if not compatible:
        raise RuntimeError(
            f'No compatible ViT parameters found in {checkpoint_path}')

    incompatible = backbone.load_state_dict(compatible, strict=False)
    print(
        f'[PRETRAINED] Loaded {len(compatible)}/{len(model_state)} tensors '
        f'from {checkpoint_path}',
        flush=True,
    )
    if incompatible.missing_keys:
        print(
            f'[PRETRAINED] Missing tensors: {len(incompatible.missing_keys)} '
            '(the classification head may differ, which is expected)',
            flush=True,
        )


def run_masked_vit_blocks(blocks, hidden, valid_tokens):
    """Exclude padded keys from attention and zero padded query positions."""
    attention_mask = valid_tokens[:, None, None, :]
    query_mask = valid_tokens.unsqueeze(-1).to(hidden.dtype)
    for block in blocks:
        hidden = block(hidden, attn_mask=attention_mask)
        hidden = hidden * query_mask
    return hidden


# ==============================================================================
# APT Patch Merge ViT Model
# ==============================================================================

class APTPatchMergeViT(nn.Module):
    """
    ViT-B/16 with APT-style multi-scale patch merging.

    对每张图:
      1. 计算 16×16 和 32×32 两个尺度的 patch 熵值
      2. 按 2×2 block 遍历 (14×14 grid → 7×7 blocks)
      3. 如果 32×32 熵值 < threshold → 合并 4 个 sub-patch 为 1 token
      4. 如果 32×32 熵值 >= threshold → 保持 4 个独立 tokens
      5. 组装混合序列 → ViT Blocks → 分类

    Args:
        num_classes: 类别数
        merge_threshold: 32×32 patch 熵值阈值，低于此值合并
        img_size: 输入图像尺寸
    """

    def __init__(self, num_classes=100, merge_threshold=5.5,
                 img_size=224, pretrained=True, drop_path_rate=0.0,
                 input_mean=(0.485, 0.456, 0.406),
                 input_std=(0.229, 0.224, 0.225),
                 pretrained_checkpoint=None):
        super().__init__()
        self.merge_threshold = merge_threshold
        self.img_size = img_size
        self.input_mean = input_mean
        self.input_std = input_std

        backbone = timm.create_model(
            'vit_base_patch16_224.augreg_in21k',
            pretrained=False if pretrained_checkpoint else pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            img_size=img_size,
        )
        if pretrained_checkpoint:
            load_local_pretrained(backbone, pretrained_checkpoint)

        self.patch_embed = backbone.patch_embed
        self.cls_token = backbone.cls_token
        self.pos_embed = backbone.pos_embed   # (1, 197, 768)
        self.pos_drop = backbone.pos_drop
        self.blocks = backbone.blocks
        self.norm = backbone.norm
        self.head = backbone.head

        self.num_patches = self.patch_embed.num_patches  # 196
        self.grid_size = self.patch_embed.grid_size       # (14, 14)
        self.embed_dim = backbone.embed_dim               # 768

        # Precompute block mapping: 49 blocks of 2×2 sub-patches → 196 input indices
        # block_sub[k] = 4 linear indices of the 2×2 sub-patches
        block_sub = []
        for i in range(7):
            for j in range(7):
                base = i * 28 + j * 2  # row i*2, col j*2 in 14×14 grid
                block_sub.append([base, base + 1, base + 14, base + 15])
        self.register_buffer('block_sub_idx', torch.tensor(block_sub, dtype=torch.long))  # (49, 4)

        # Precompute 7×7 position embeddings (for merged 32×32 tokens)
        with torch.no_grad():
            patch_pos = self.pos_embed[:, 1:, :].clone()  # (1, 196, 768)
            patch_pos_3d = patch_pos.reshape(1, 14, 14, self.embed_dim).permute(0, 3, 1, 2)
            merged_pos_3d = F.interpolate(patch_pos_3d, size=(7, 7),
                                          mode='bicubic', align_corners=False)
            merged = merged_pos_3d.permute(0, 2, 3, 1).reshape(1, 49, self.embed_dim).clone()
        self.register_buffer('merged_pos_embed', merged.squeeze(0))  # (49, D)

        # For logging
        self._last_k = self.num_patches
        self._last_n = self.num_patches
        self._last_merged = 0
        self._last_token_counts = None

        del backbone

    def forward(self, x):
        B = x.shape[0]

        # Step 1: Unnormalize and compute only the 32x32 entropy used for merging.
        mean = x.new_tensor(self.input_mean).view(1, 3, 1, 1)
        std = x.new_tensor(self.input_std).view(1, 3, 1, 1)
        images_255 = ((x * std + mean) * 255.0).clamp(0, 255)

        entropy_maps = compute_patch_entropy_fast(
            images_255, patch_size=32, num_scales=1)
        entropy32 = entropy_maps[32]  # (B, 7, 7)

        # Step 2: Patch embedding (ALL 196 patches)
        x_patches = self.patch_embed(x)  # (B, 196, 768)
        D = self.embed_dim

        # Step 3: Vectorized merge decision
        merge_mask = (entropy32 < self.merge_threshold).reshape(B, 49)  # (B, 49)
        num_merged = merge_mask.sum(dim=1).long()  # (B,)
        K_per_image = 196 - 3 * num_merged  # (B,) — output tokens per image
        # Step 4: Build all 49x4 candidates, then compact valid entries.
        block_sub = self.block_sub_idx
        sub_tokens = x_patches[:, block_sub]  # (B, 49, 4, D)
        merged_tokens = sub_tokens.mean(dim=2)
        pos16 = self.pos_embed[0, 1:][block_sub]  # (49, 4, D)

        first_tokens = torch.where(
            merge_mask.unsqueeze(-1),
            merged_tokens,
            sub_tokens[:, :, 0],
        )
        candidates = torch.cat(
            [first_tokens.unsqueeze(2), sub_tokens[:, :, 1:]], dim=2)
        first_positions = torch.where(
            merge_mask.unsqueeze(-1),
            self.merged_pos_embed.unsqueeze(0),
            pos16[:, 0].unsqueeze(0),
        )
        candidate_positions = torch.cat([
            first_positions.unsqueeze(2),
            pos16[:, 1:].unsqueeze(0).expand(B, -1, -1, -1),
        ], dim=2)
        candidate_valid = torch.cat([
            torch.ones_like(merge_mask).unsqueeze(-1),
            (~merge_mask).unsqueeze(-1).expand(-1, -1, 3),
        ], dim=2)

        candidates = candidates.flatten(1, 2)
        candidate_positions = candidate_positions.flatten(1, 2)
        candidate_valid = candidate_valid.flatten(1)
        slot_count = candidate_valid.shape[1]
        slot_indices = torch.arange(slot_count, device=x.device)
        sort_keys = slot_indices.unsqueeze(0) + (~candidate_valid) * slot_count
        order = sort_keys.argsort(dim=1)
        gather_index = order.unsqueeze(-1).expand(-1, -1, D)
        candidates = candidates.gather(1, gather_index)
        candidate_positions = candidate_positions.gather(1, gather_index)

        K_max = int(K_per_image.max().item())
        tokens_out = candidates[:, :K_max]
        pos_out = candidate_positions[:, :K_max]
        valid_mask = (
            torch.arange(K_max, device=x.device).unsqueeze(0)
            < K_per_image.unsqueeze(1)
        )

        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_pos = self.pos_embed[:, 0:1, :].expand(B, -1, -1)
        h = torch.cat([cls_tokens, tokens_out], dim=1)
        pos = torch.cat([cls_pos, pos_out], dim=1)
        h = h + pos
        h = self.pos_drop(h)

        # Step 5: Transformer blocks with padded keys excluded from attention.
        sequence_valid = torch.cat([
            torch.ones(B, 1, dtype=torch.bool, device=x.device),
            valid_mask,
        ], dim=1)
        h = run_masked_vit_blocks(self.blocks, h, sequence_valid)

        h = self.norm(h)
        logits = self.head(h[:, 0])

        self._last_k = K_max
        self._last_n = self.num_patches
        self._last_merged = num_merged.float().mean()
        self._last_token_counts = K_per_image

        return logits


# ==============================================================================
# Training Utilities (same as train_apt_patch_selection.py)
# ==============================================================================

DATASETS = {
    'cifar100':    (get_cifar100_loader,     100, 100),
    'oxford_pets': (get_oxford_pets_loader,   37, 100),
    'food101':     (get_food101_loader,      101,  30),
}

BASELINE_ACC = {
    'cifar100':   91.69,
    'oxford_pets': 93.81,
    'food101':    91.37,
}


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write('\n')


def atomic_torch_save(payload, path):
    """Avoid leaving a corrupt checkpoint at the final resume path."""
    temporary_path = f'{path}.tmp'
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def train_one_epoch(model, loader, criterion, optimizer, device,
                    accum_steps=1, epoch=None, total_epochs=None,
                    use_amp=True, log_interval=20):
    model.train()
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = 0
    optimizer.zero_grad(set_to_none=True)

    desc = f'Epoch {epoch}/{total_epochs}' if epoch is not None else 'Train'
    pbar = tqdm(enumerate(loader), total=len(loader), desc=desc, unit='batch',
                dynamic_ncols=True)

    for batch_idx, (images, targets) in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(
                device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            logits = model(images)
            full_loss = criterion(logits, targets)
            loss = full_loss / accum_steps
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += full_loss.detach()
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum()

        if (batch_idx + 1) % log_interval == 0:
            postfix = {
                'loss': f'{total_loss.item() / (batch_idx + 1):.4f}',
                'acc': f'{100.0 * correct.item() / total:.1f}%',
                'tokens': f'{model._last_k}/{model._last_n}',
            }
            pbar.set_postfix(postfix)

    if (batch_idx + 1) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    n_batches = len(loader)
    return (
        total_loss.item() / n_batches,
        100.0 * correct.item() / total,
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device, track_patches=False, desc='Eval',
             use_amp=True, log_interval=20):
    model.eval()
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = 0
    token_sum = torch.zeros((), device=device)
    padded_token_sum = 0
    token_samples = 0
    pbar = tqdm(loader, total=len(loader), desc=desc, unit='batch', dynamic_ncols=True)
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(
                device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        total_loss += loss
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum()
        if track_patches:
            counts = model._last_token_counts
            token_sum += counts.sum()
            padded_token_sum += model._last_k * counts.numel()
            token_samples += counts.numel()
        if (batch_idx + 1) % log_interval == 0:
            pbar.set_postfix({
                'loss': f'{total_loss.item() / (batch_idx + 1):.4f}',
                'acc': f'{100.0 * correct.item() / total:.1f}%',
            })

    result = (
        total_loss.item() / len(loader),
        100.0 * correct.item() / total,
    )
    if track_patches and token_samples:
        avg_k = token_sum.item() / token_samples
        avg_padded = padded_token_sum / token_samples
        avg_n = model._last_n
        result += (avg_k, avg_n, avg_k / avg_n * 100, avg_padded)
    return result


@torch.no_grad()
def compute_efficiency_metrics(model, loader, device, use_amp=True):
    model.eval()
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(
                device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            _ = model(images)
        break

    if device != 'cpu':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    total_time = 0.0
    total_samples = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        batch_size = images.size(0)
        if device != 'cpu':
            torch.cuda.synchronize()
        start = time.time()
        with torch.autocast(
                device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            _ = model(images)
        if device != 'cpu':
            torch.cuda.synchronize()
        total_time += time.time() - start
        total_samples += batch_size

    latency = total_time / total_samples * 1000
    throughput = total_samples / total_time
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if device != 'cpu' else 0.0
    )
    return latency, throughput, peak_memory_mb


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='APT-style multi-scale patch merge for ViT')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASETS.keys()))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=5.5,
                        help='32x32 entropy threshold (below → merge 4→1)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accum', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override default epoch count')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--pretrained_checkpoint',
        default=None,
        help='Local ViT-B/16 checkpoint; disables online weight download',
    )
    parser.add_argument('--no_pretrained', action='store_true')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', type=str, default=None,
                        help='Only evaluate using the given checkpoint path')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu' and args.eval_only is None:
        raise RuntimeError('CUDA is required for formal training')
    use_amp = device != 'cpu' and not args.no_amp
    print(f'Device: {device}', flush=True)
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(args.gpu)}', flush=True)

    loader_fn, num_classes, epochs = DATASETS[args.dataset]
    if args.epochs is not None:
        epochs = args.epochs
    effective_bs = args.batch_size * args.accum

    print(f'Dataset: {args.dataset}', flush=True)
    print(f'  Batch: {args.batch_size}, Accum: {args.accum}, Effective: {effective_bs}', flush=True)
    print(f'  Epochs: {epochs}, LR: {args.lr}, WD: {args.weight_decay}', flush=True)
    print(f'  Label smoothing: {args.label_smoothing}', flush=True)
    print(f'  Merge threshold (32x32): {args.threshold}', flush=True)
    print(f'  Baseline Acc: {BASELINE_ACC[args.dataset]:.2f}%', flush=True)

    loader_kwargs = {
        'batch_size': args.batch_size,
        'data_dir': './data',
        'num_workers': args.num_workers,
        'image_size': args.image_size,
    }
    if args.dataset == 'cifar100':
        loader_kwargs['return_val'] = True
    result = loader_fn(**loader_kwargs)
    if len(result) == 4:
        train_loader, val_loader, test_loader, n_cls = result
        val_is_test = False
        print(f'  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, '
              f'Test: {len(test_loader.dataset)}, Classes: {n_cls}', flush=True)
    else:
        train_loader, test_loader, n_cls = result
        val_loader = test_loader
        val_is_test = True
        print(f'  Train: {len(train_loader.dataset)}, Test/Val: {len(test_loader.dataset)}, '
              f'Classes: {n_cls}', flush=True)

    print('Creating APT Patch Merge ViT-B/16...', flush=True)
    model = APTPatchMergeViT(
        num_classes=n_cls,
        merge_threshold=args.threshold,
        img_size=args.image_size,
        pretrained=not args.no_pretrained,
        input_mean=NORMALIZATION[args.dataset][0],
        input_std=NORMALIZATION[args.dataset][1],
        pretrained_checkpoint=args.pretrained_checkpoint,
    )
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'  Total params: {total_params:.2f}M', flush=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=device != 'cpu',
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_epoch = -1
    start_epoch = 0
    save_dir = (
        f'./checkpoints/{args.dataset}_apt_merge_t{args.threshold}_s{args.seed}'
    )
    os.makedirs(save_dir, exist_ok=True)
    latest_path = os.path.join(save_dir, 'checkpoint_latest.pth')
    best_path = os.path.join(save_dir, 'best_model.pth')
    history_path = os.path.join(save_dir, 'history.json')
    results_path = os.path.join(save_dir, 'results.json')

    # ---- eval_only mode ----
    if args.eval_only is not None:
        ckpt_path = args.eval_only
        print(f'[EVAL ONLY] Loading: {ckpt_path}', flush=True)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'  Epoch: {ckpt.get("epoch", "?")}, Val Acc: {ckpt.get("val_acc", "?"):.2f}%', flush=True)

        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, desc='Test',
            use_amp=use_amp, log_interval=args.log_interval)
        val_loss, val_acc, avg_k, avg_n, keep_pct, avg_padded = evaluate(
            model, val_loader, criterion, device, track_patches=True,
            desc='Val', use_amp=use_amp, log_interval=args.log_interval)

        print('\n=== Efficiency Metrics ===', flush=True)
        latency, throughput, peak_memory_mb = compute_efficiency_metrics(
            model, test_loader, device, use_amp=use_amp)

        baseline_acc = BASELINE_ACC[args.dataset]
        print(f'\n{"="*20} Evaluation ({args.dataset}) {"="*20}', flush=True)
        print(f'  Val Acc:        {val_acc:.2f}%', flush=True)
        print(f'  Test Acc:       {test_acc:.2f}%', flush=True)
        print(f'  Baseline Acc:   {baseline_acc:.2f}%', flush=True)
        print(f'  Acc Diff:       {test_acc - baseline_acc:+.2f}%', flush=True)
        print(f'  Real tokens:    {avg_k:.1f}/{int(avg_n)} ({keep_pct:.1f}%)', flush=True)
        print(f'  Padded tokens:  {avg_padded:.1f}/{int(avg_n)}', flush=True)
        print(f'  Threshold:      {args.threshold}', flush=True)
        print(f'  Latency:        {latency:.2f} ms/sample', flush=True)
        print(f'  Throughput:     {throughput:.2f} samples/sec', flush=True)
        print(f'  Peak memory:    {peak_memory_mb:.1f} MiB', flush=True)
        print(f'{"="*58}\n', flush=True)
        return

    # ---- Resume from checkpoint ----
    ckpt_path = args.resume
    if ckpt_path is None and os.path.isfile(latest_path):
        ckpt_path = latest_path
        print(f'[AUTO RESUME] Found: {ckpt_path}', flush=True)
    elif ckpt_path is not None:
        print(f'[RESUME] Loading: {ckpt_path}', flush=True)

    if ckpt_path is not None and os.path.isfile(ckpt_path):
        ckpt = torch.load(
            ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        best_epoch = ckpt.get('best_epoch', -1)
        print(f'  Resumed from epoch {ckpt["epoch"] + 1}, Best Val: {best_val_acc:.2f}%', flush=True)

    write_json(os.path.join(save_dir, 'args.json'), vars(args))
    history = []
    if os.path.isfile(history_path):
        with open(history_path, encoding='utf-8') as file:
            history = json.load(file)

    print(f'\n{"="*60}', flush=True)
    print(f'APT Patch Merge ViT-B/16 on {args.dataset} ({epochs} epochs)', flush=True)
    if start_epoch > 0:
        print(f'Resuming from epoch {start_epoch + 1}', flush=True)
    print(f'{"="*60}\n', flush=True)

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            args.accum, epoch=epoch + 1, total_epochs=epochs,
            use_amp=use_amp, log_interval=args.log_interval)
        scheduler.step()

        val_loss, val_acc, avg_k, avg_n, keep_pct, avg_padded = evaluate(
            model, val_loader, criterion, device, track_patches=True,
            desc='Val', use_amp=use_amp, log_interval=args.log_interval)

        print(f'Epoch {epoch+1}/{epochs}')
        print(f'  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        print(f'  Real tokens: {avg_k:.1f}/{int(avg_n)} ({keep_pct:.1f}%)')
        print(f'  Padded tokens: {avg_padded:.1f}/{int(avg_n)}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'args': vars(args),
            'avg_tokens': avg_k,
            'avg_padded_tokens': avg_padded,
            'avg_total': int(avg_n),
            'keep_pct': keep_pct,
        }
        atomic_torch_save(checkpoint_data, latest_path)

        history = [
            item for item in history if item.get('epoch') != epoch + 1
        ]
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'avg_real_tokens': avg_k,
            'avg_padded_tokens': avg_padded,
            'original_tokens': int(avg_n),
            'token_ratio': avg_k / avg_n,
            'epoch_seconds': time.time() - epoch_start,
        })
        history.sort(key=lambda item: item['epoch'])
        write_json(history_path, history)

        if improved:
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
                'best_epoch': best_epoch,
                'args': vars(args),
                'avg_tokens': avg_k,
                'avg_padded_tokens': avg_padded,
                'avg_total': int(avg_n),
                'keep_pct': keep_pct,
            }
            atomic_torch_save(best_checkpoint, best_path)
            print(f'  -> Saved best (Val: {best_val_acc:.2f}%)')
        print(flush=True)

    # ---- Final evaluation ----
    print('\n=== Evaluating best model on test set ===', flush=True)
    if not os.path.isfile(best_path):
        raise RuntimeError(f'Best model not found: {best_path}')
    best_checkpoint = torch.load(
        best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    best_val_acc = best_checkpoint['best_val_acc']
    best_epoch = best_checkpoint['best_epoch']
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device, desc='Test',
        use_amp=use_amp, log_interval=args.log_interval)

    print('\n=== Efficiency Metrics ===', flush=True)
    latency, throughput, peak_memory_mb = compute_efficiency_metrics(
        model, test_loader, device, use_amp=use_amp)

    baseline_acc = BASELINE_ACC[args.dataset]
    acc_diff = test_acc - baseline_acc
    result_payload = {
        'method': 'fixed_16_32_patch_merge',
        'dataset': args.dataset,
        'seed': args.seed,
        'epochs': epochs,
        'best_epoch': best_epoch + 1,
        'best_val_acc': best_val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'baseline_acc': baseline_acc,
        'acc_diff': acc_diff,
        'threshold32': args.threshold,
        'original_tokens': int(best_checkpoint['avg_total']),
        'avg_real_tokens': best_checkpoint['avg_tokens'],
        'avg_padded_tokens': best_checkpoint['avg_padded_tokens'],
        'token_ratio': (
            best_checkpoint['avg_tokens'] / best_checkpoint['avg_total']
        ),
        'latency_ms': latency,
        'throughput': throughput,
        'peak_memory_mb': peak_memory_mb,
    }
    write_json(results_path, result_payload)

    print(f'\n{"="*20} Final Results ({args.dataset}) {"="*20}', flush=True)
    print(f'  Best Val Epoch: {best_epoch+1}', flush=True)
    print(f'  Best Val Acc:   {best_val_acc:.2f}%', flush=True)
    print(f'  Test Acc:       {test_acc:.2f}%', flush=True)
    print(f'  Baseline Acc:   {baseline_acc:.2f}%', flush=True)
    print(f'  Acc Diff:       {acc_diff:+.2f}%', flush=True)
    print(f'  Threshold:      {args.threshold}', flush=True)
    print(f'  Latency:         {latency:.2f} ms/sample', flush=True)
    print(f'  Throughput:      {throughput:.2f} samples/sec', flush=True)
    print(f'  Peak memory:     {peak_memory_mb:.1f} MiB', flush=True)
    print(f'  Results:         {results_path}', flush=True)
    print(f'{"="*58}\n', flush=True)


if __name__ == '__main__':
    main()
