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
  python apt_experiments/train_apt_patch_merge.py --dataset cifar100 --gpu 0
  python apt_experiments/train_apt_patch_merge.py --dataset oxford_pets --gpu 0 --threshold 5.0

  # 仅评估
  python apt_experiments/train_apt_patch_merge.py --dataset cifar100 --gpu 0 --eval_only PATH
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import sys
import random
import argparse
import json
import numpy as np

APT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APT_ROOT)
sys.path.insert(0, APT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
from datasets import (
    get_cifar100_loader,
    get_dtd_loader,
    get_food101_loader,
    get_oxford_pets_loader,
)
from apt_utils import (
    TokenStatsAccumulator,
    compute_patch_entropy,
    denormalize_to_255,
    get_normalization,
    run_masked_vit_blocks,
    write_result_json,
)
from tqdm import tqdm
import timm


# ==============================================================================
# APT Entropy Computation (shared with train_apt_patch_selection.py)
# ==============================================================================

def compute_patch_entropy_fast(images_255, patch_size=16, num_scales=2, bins=64):
    """Compatibility wrapper around the shared entropy implementation."""
    patch_sizes = [patch_size * (2 ** i) for i in range(num_scales)]
    return compute_patch_entropy(images_255, patch_sizes=patch_sizes, bins=bins)


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
                 entropy_bins=64,
                 backbone_name='vit_base_patch16_224.augreg_in21k'):
        super().__init__()
        self.merge_threshold = merge_threshold
        self.img_size = img_size
        self.input_mean = input_mean
        self.input_std = input_std
        self.entropy_bins = entropy_bins

        backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            img_size=img_size,
        )

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
        B, C, H, W = x.shape

        # Step 1: Unnormalize and compute entropy at two scales
        images_255 = denormalize_to_255(x, self.input_mean, self.input_std)

        entropy_maps = compute_patch_entropy_fast(
            images_255, patch_size=16, num_scales=2, bins=self.entropy_bins)
        entropy32 = entropy_maps[32]  # (B, 7, 7)

        # Step 2: Patch embedding (ALL 196 patches)
        x_patches = self.patch_embed(x)  # (B, 196, 768)
        D = self.embed_dim

        # Step 3: Vectorized merge decision
        merge_mask = (entropy32 < self.merge_threshold).reshape(B, 49)  # (B, 49)
        num_merged = merge_mask.sum(dim=1).long()  # (B,)
        K_per_image = 196 - 3 * num_merged  # (B,) — output tokens per image
        K_max = int(K_per_image.max().item())
        K_min = int(K_per_image.min().item())

        # Allocate output: worst-case max tokens in batch
        tokens_out = torch.zeros(B, K_max, D, device=x.device,
                                 dtype=x_patches.dtype)
        pos_out = torch.zeros(B, K_max, D, device=x.device,
                              dtype=x_patches.dtype)

        # Precompute patch16 position embed (offset +1 to skip CLS token)
        pos16_all = self.pos_embed[0, 1:]  # (196, D)

        # Per-image write cursors (tracked on CPU then indexed)
        write_pos = torch.zeros(B, dtype=torch.long, device=x.device)

        # Process 49 blocks in a single loop with batched ops
        block_sub = self.block_sub_idx  # (49, 4)
        for k in range(49):
            sub_idx = block_sub[k]      # (4,)
            merge = merge_mask[:, k]    # (B,) bool

            # Gather sub-patch tokens: (B, 4, D)
            sub_tokens = x_patches[:, sub_idx]
            # Merged token: (B, D)
            merged = sub_tokens.mean(dim=1)
            # 16×16 pos for these 4 sub-patches: (4, D)
            pos4 = pos16_all[sub_idx]

            # ---- Images that KEEP (merge=False) ----
            keep_idx = (~merge).nonzero(as_tuple=True)[0]  # (N_keep,)
            if keep_idx.numel() > 0:
                wp = write_pos[keep_idx]  # (N_keep,)
                for jj in range(4):
                    tokens_out[keep_idx, wp + jj] = sub_tokens[keep_idx, jj]
                    pos_out[keep_idx, wp + jj] = pos4[jj].unsqueeze(0)
                write_pos[keep_idx] += 4

            # ---- Images that MERGE (merge=True) ----
            merge_idx = (merge).nonzero(as_tuple=True)[0]  # (N_merge,)
            if merge_idx.numel() > 0:
                wp = write_pos[merge_idx]  # (N_merge,)
                tokens_out[merge_idx, wp] = merged[merge_idx]
                pos_out[merge_idx, wp] = self.merged_pos_embed[k].unsqueeze(0)
                write_pos[merge_idx] += 1

        # Step 4: Build valid mask, add CLS token
        valid_mask = torch.zeros(B, K_max, dtype=torch.bool, device=x.device)
        for b in range(B):
            valid_mask[b, :K_per_image[b]] = True

        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_pos = self.pos_embed[:, 0:1, :].expand(B, -1, -1)
        h = torch.cat([cls_tokens, tokens_out], dim=1)
        pos = torch.cat([cls_pos, pos_out], dim=1)
        h = h + pos
        h = self.pos_drop(h)

        # Step 5: Transformer blocks — zero out padded positions
        pad_mask = torch.cat([
            torch.ones(B, 1, dtype=torch.bool, device=x.device),
            valid_mask,
        ], dim=1).unsqueeze(-1)  # (B, 1+K_max, 1)

        h = run_masked_vit_blocks(
            self.blocks, h, pad_mask.squeeze(-1))

        h = self.norm(h)
        logits = self.head(h[:, 0])

        self._last_k = K_max
        self._last_n = self.num_patches
        self._last_merged = num_merged.float().mean().item()
        self._last_token_counts = K_per_image.detach()

        return logits


# ==============================================================================
# Training Utilities (same as train_apt_patch_selection.py)
# ==============================================================================

DATASETS = {
    'cifar100':    (get_cifar100_loader,     100, 100),
    'oxford_pets': (get_oxford_pets_loader,   37, 100),
    'food101':     (get_food101_loader,      101,  30),
    'dtd':         (get_dtd_loader,            47, 100),
}

BASELINE_ACC = {
    'cifar100':   91.69,
    'oxford_pets': 93.81,
    'food101':    91.37,
    'dtd':        80.85,
}


def train_one_epoch(model, loader, criterion, optimizer, device,
                    accum_steps=1, epoch=None, total_epochs=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    desc = f'Epoch {epoch}/{total_epochs}' if epoch is not None else 'Train'
    pbar = tqdm(enumerate(loader), total=len(loader), desc=desc, unit='batch',
                dynamic_ncols=True)

    for batch_idx, (images, targets) in pbar:
        images, targets = images.to(device), targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)
        loss = loss / accum_steps
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        current_acc = 100.0 * correct / total
        current_loss = total_loss / (batch_idx + 1)
        postfix = {'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.1f}%'}
        if hasattr(model, '_last_k'):
            merge_info = f'{model._last_merged:.0f}' if hasattr(model, '_last_merged') else '?'
            postfix['tokens'] = f'{model._last_k}/{model._last_n}(m{merge_info})'
        pbar.set_postfix(postfix)

    if (batch_idx + 1) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    n_batches = len(loader)
    return total_loss / n_batches, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, track_patches=False, desc='Eval'):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    token_stats = TokenStatsAccumulator() if track_patches else None
    pbar = tqdm(loader, total=len(loader), desc=desc, unit='batch', dynamic_ncols=True)
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if track_patches:
            token_stats.update(model._last_token_counts, model._last_k)
        pbar.set_postfix({
            'loss': f'{total_loss / (pbar.n if pbar.n else 1):.4f}',
            'acc': f'{100.0 * correct / total:.1f}%',
        })

    result = (total_loss / len(loader), 100.0 * correct / total)
    if track_patches:
        stats = token_stats.compute()
        avg_k = stats.mean
        avg_n = model._last_n
        result = result + (avg_k, avg_n, avg_k / avg_n * 100, stats)
    return result


@torch.no_grad()
def compute_efficiency_metrics(model, loader, device):
    model.eval()
    for images, _ in loader:
        images = images.to(device)
        _ = model(images)
        break

    total_time = 0.0
    total_samples = 0
    for images, _ in loader:
        images = images.to(device)
        batch_size = images.size(0)
        if device != 'cpu':
            torch.cuda.synchronize()
        start = time.time()
        _ = model(images)
        if device != 'cpu':
            torch.cuda.synchronize()
        total_time += time.time() - start
        total_samples += batch_size

    latency = total_time / total_samples * 1000
    throughput = total_samples / total_time
    return latency, throughput


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
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override default epoch count')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--entropy_bins', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
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

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}', flush=True)
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(args.gpu)}', flush=True)
    elif args.eval_only is None:
        raise RuntimeError(
            'CUDA is required for training. CPU is limited to direct model smoke tests.'
        )

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

    loader_kwargs = dict(batch_size=args.batch_size,
                         data_dir=os.path.join(PROJECT_ROOT, 'data'),
                         num_workers=4, image_size=args.image_size)
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
        input_mean=get_normalization(args.dataset)[0],
        input_std=get_normalization(args.dataset)[1],
        entropy_bins=args.entropy_bins,
    )
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'  Total params: {total_params:.2f}M', flush=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_epoch = -1
    start_epoch = 0
    save_dir = (
        os.path.join(
            APT_ROOT,
            'checkpoints',
            f'{args.dataset}_apt_merge_t{args.threshold}_s{args.seed}',
        )
    )
    os.makedirs(save_dir, exist_ok=True)

    # ---- eval_only mode ----
    if args.eval_only is not None:
        ckpt_path = args.eval_only
        print(f'[EVAL ONLY] Loading: {ckpt_path}', flush=True)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'  Epoch: {ckpt.get("epoch", "?")}, Val Acc: {ckpt.get("val_acc", "?"):.2f}%', flush=True)

        test_loss, test_acc = evaluate(model, test_loader, criterion, device, desc='Test')
        val_loss, val_acc, avg_k, avg_n, keep_pct, token_stats = evaluate(
            model, val_loader, criterion, device, track_patches=True, desc='Val')

        print('\n=== Efficiency Metrics ===', flush=True)
        latency, throughput = compute_efficiency_metrics(model, test_loader, device)

        baseline_acc = BASELINE_ACC[args.dataset]
        print(f'\n{"="*20} Evaluation ({args.dataset}) {"="*20}', flush=True)
        print(f'  Val Acc:        {val_acc:.2f}%', flush=True)
        print(f'  Test Acc:       {test_acc:.2f}%', flush=True)
        print(f'  Baseline Acc:   {baseline_acc:.2f}%', flush=True)
        print(f'  Acc Diff:       {test_acc - baseline_acc:+.2f}%', flush=True)
        print(f'  Tokens:         {int(avg_k)}/{int(avg_n)} ({keep_pct:.1f}%)', flush=True)
        print(f'  Token P50/P90:  {token_stats.p50:.0f}/{token_stats.p90:.0f}', flush=True)
        print(f'  Threshold:      {args.threshold}', flush=True)
        print(f'  Latency:        {latency:.2f} ms/sample', flush=True)
        print(f'  Throughput:     {throughput:.2f} samples/sec', flush=True)
        print(f'{"="*58}\n', flush=True)
        return

    # ---- Resume from checkpoint ----
    ckpt_path = None
    if args.resume is not None:
        ckpt_path = args.resume
        print(f'[RESUME] Loading: {ckpt_path}', flush=True)
    else:
        epoch_files = sorted(
            [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch_')],
            key=lambda f: int(f.split('_')[-1].split('.')[0]))
        if epoch_files:
            ckpt_path = os.path.join(save_dir, epoch_files[-1])
            print(f'[AUTO RESUME] Found: {ckpt_path}', flush=True)

    if ckpt_path is not None and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        best_epoch = ckpt.get('best_epoch', -1)
        print(f'  Resumed from epoch {ckpt["epoch"] + 1}, Best Val: {best_val_acc:.2f}%', flush=True)

    write_result_json(os.path.join(save_dir, 'args.json'), vars(args))

    print(f'\n{"="*60}', flush=True)
    print(f'APT Patch Merge ViT-B/16 on {args.dataset} ({epochs} epochs)', flush=True)
    if start_epoch > 0:
        print(f'Resuming from epoch {start_epoch + 1}', flush=True)
    print(f'{"="*60}\n', flush=True)

    if start_epoch >= epochs:
        print('All epochs already completed.', flush=True)
        return

    for epoch in range(start_epoch, epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            args.accum, epoch=epoch + 1, total_epochs=epochs)
        scheduler.step()

        val_result = evaluate(
            model, val_loader, criterion, device, track_patches=True, desc='Val')
        val_loss, val_acc, avg_k, avg_n, keep_pct, token_stats = val_result

        print(f'Epoch {epoch+1}/{epochs}')
        print(f'  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        print(f'  Tokens: {int(avg_k)}/{int(avg_n)} ({keep_pct:.1f}%)')
        print(f'  Token P50/P90: {token_stats.p50:.0f}/{token_stats.p90:.0f}, '
              f'padded mean: {token_stats.padded_mean:.1f}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'args': vars(args),
            'avg_tokens': int(avg_k),
            'avg_total': int(avg_n),
            'keep_pct': keep_pct,
            'token_stats': token_stats.to_dict(),
        }
        torch.save(checkpoint_data, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            checkpoint_data['test_acc'] = None
            torch.save(checkpoint_data, f'{save_dir}/best_model.pth')
            print(f'  -> Saved best validation model ({best_val_acc:.2f}%)')
        print(flush=True)

    # ---- Final evaluation ----
    print('\n=== Evaluating frozen best model on test set ===', flush=True)
    best_path = f'{save_dir}/best_model.pth'
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, desc='Test')

    print('\n=== Efficiency Metrics ===', flush=True)
    latency, throughput = compute_efficiency_metrics(model, test_loader, device)

    baseline_acc = BASELINE_ACC[args.dataset]
    acc_diff = test_acc - baseline_acc

    print(f'\n{"="*20} Final Results ({args.dataset}) {"="*20}', flush=True)
    print(f'  Best Val Epoch: {best_epoch+1}', flush=True)
    print(f'  Best Val Acc:   {best_val_acc:.2f}%', flush=True)
    print(f'  Test Acc:       {test_acc:.2f}%', flush=True)
    print(f'  Baseline Acc:   {baseline_acc:.2f}%', flush=True)
    print(f'  Acc Diff:       {acc_diff:+.2f}%', flush=True)
    print(f'  Threshold:      {args.threshold}', flush=True)
    print(f'  Latency:         {latency:.2f} ms/sample', flush=True)
    print(f'  Throughput:      {throughput:.2f} samples/sec', flush=True)
    print(f'{"="*58}\n', flush=True)
    write_result_json(os.path.join(save_dir, 'results.json'), {
        'method': 'apt_merge',
        'dataset': args.dataset,
        'seed': args.seed,
        'threshold': args.threshold,
        'entropy_bins': args.entropy_bins,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'baseline_acc': baseline_acc,
        'acc_diff': acc_diff,
        'latency_ms': latency,
        'throughput': throughput,
        'args': vars(args),
    })


if __name__ == '__main__':
    main()
