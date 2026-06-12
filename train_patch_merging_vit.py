"""
Patch Merging-ViT: Router-guided multi-scale patch merging for ViT acceleration.

与 APT Patch Merge 的区别：
  - APT: 基于熵值阈值决定合并/保留（不可学习，需手动调参）
  - Patch Merging-ViT: 基于 Router 学习决定合并/保留（可学习，自适应数据集）

Architecture:
  Image → Patch Embed (196 patches)
    → Router (MLP) → 49 个 2×2 block 的重要性分数
    → Top-K 选择 → 保留 K 个 block (4 tokens each)
                  → 合并 (49-K) 个 block (4→1 token via avg pool)
    → 混合序列 [CLS, kept, merged] → 12 × ViT Block → 分类

Token 数量公式: 3K + 49 (K = 保留的 block 数)

Usage:
  # 训练
  python train_patch_merging_vit.py --dataset cifar100 --gpu 0 --keep_ratio 0.75

  # 评估
  python train_patch_merging_vit.py --dataset cifar100 --gpu 0 --eval_only ./checkpoints/xxx/best_model.pth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import get_cifar100_loader, get_oxford_pets_loader, get_food101_loader
from tqdm import tqdm
import timm


# ==============================================================================
# Patch Merging ViT Model
# ==============================================================================

class PatchMergingViT(nn.Module):
    """
    ViT-B/16 with Router-guided patch merging.

    对每张图:
      1. Patch Embedding → 196 patches (B, 196, 768)
      2. 划分为 49 个 2×2 block
      3. Router 对每个 block 打分 (重要性分数)
      4. Top-K 选择保留 K 个 block (保持 4 个独立 token)
      5. 其余 block 合并为 1 个 token (平均池化)
      6. 组装混合序列 → 12 × ViT Block → 分类

    Args:
        num_classes: 类别数
        keep_ratio: 保留 block 比例 (0.75 = 保留 37/49 个 block)
        img_size: 输入图像尺寸
        pretrained: 是否加载预训练权重
        drop_path_rate: DropPath 率
    """

    def __init__(self, num_classes=100, keep_ratio=0.75,
                 img_size=224, pretrained=True, drop_path_rate=0.0):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.img_size = img_size

        # 1. 加载 ViT-B/16 backbone
        backbone = timm.create_model(
            'vit_base_patch16_224.augreg_in21k',
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            img_size=img_size,
        )

        # 2. 提取组件
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

        # 3. Router: 输入 block 平均特征，输出重要性分数
        self.router = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Linear(self.embed_dim // 2, 1),
        )

        # 4. 预计算 block 映射: 49 个 2×2 block → 196 patch 索引
        block_sub = []
        for i in range(7):
            for j in range(7):
                base = i * 28 + j * 2  # row i*2, col j*2 in 14×14 grid
                block_sub.append([base, base + 1, base + 14, base + 15])
        self.register_buffer('block_sub_idx', torch.tensor(block_sub, dtype=torch.long))  # (49, 4)

        # 5. 预计算 7×7 merged position embeddings (bicubic 插值)
        with torch.no_grad():
            patch_pos = self.pos_embed[:, 1:, :].clone()  # (1, 196, 768)
            patch_pos_3d = patch_pos.reshape(1, 14, 14, self.embed_dim).permute(0, 3, 1, 2)
            merged_pos_3d = F.interpolate(patch_pos_3d, size=(7, 7),
                                          mode='bicubic', align_corners=False)
            merged = merged_pos_3d.permute(0, 2, 3, 1).reshape(1, 49, self.embed_dim).clone()
        self.register_buffer('merged_pos_embed', merged.squeeze(0))  # (49, D)

        # 6. 保留的 block 数量
        self.num_keep = max(1, int(49 * keep_ratio))

        # For logging
        self._last_k = self.num_patches
        self._last_n = self.num_patches
        self._last_merged = 0

        del backbone

    def forward(self, x):
        B, C, H, W = x.shape

        # Step 1: Patch Embedding (ALL 196 patches)
        x_patches = self.patch_embed(x)  # (B, 196, 768)
        D = self.embed_dim

        # Step 2: 计算每个 block 的平均特征
        block_sub = self.block_sub_idx  # (49, 4)
        # (B, 49, 4, 768) → mean → (B, 49, 768)
        block_features = x_patches[:, block_sub].mean(dim=2)

        # Step 3: Router 打分
        scores = self.router(block_features).squeeze(-1)  # (B, 49)

        # Step 4: Top-K 选择保留的 block
        k = self.num_keep
        _, keep_indices = torch.topk(scores, k, dim=1)  # (B, k)

        # 构建合并 mask: 1=保留, 0=合并
        keep_mask = torch.zeros(B, 49, device=x.device)
        keep_mask.scatter_(1, keep_indices, 1.0)  # (B, 49)
        merge_mask = (keep_mask == 0)  # (B, 49) bool

        # 计算 token 数量
        num_merged = merge_mask.sum(dim=1).long()  # (B,)
        K_per_image = 196 - 3 * num_merged  # (B,)
        K_max = int(K_per_image.max().item())

        # Step 5: 组装 token 序列
        tokens_out = torch.zeros(B, K_max, D, device=x.device, dtype=x_patches.dtype)
        pos_out = torch.zeros(B, K_max, D, device=x.device, dtype=x_patches.dtype)

        # Precompute patch16 position embed (offset +1 to skip CLS token)
        pos16_all = self.pos_embed[0, 1:]  # (196, D)

        # Per-image write cursors
        write_pos = torch.zeros(B, dtype=torch.long, device=x.device)

        for idx in range(49):
            sub_idx = block_sub[idx]      # (4,)
            merge = merge_mask[:, idx]    # (B,) bool

            # Gather sub-patch tokens: (B, 4, D)
            sub_tokens = x_patches[:, sub_idx]
            # Merged token: (B, D)
            merged = sub_tokens.mean(dim=1)
            # 16×16 pos for these 4 sub-patches: (4, D)
            pos4 = pos16_all[sub_idx]

            # ---- Images that KEEP (merge=False) ----
            keep_idx = (~merge).nonzero(as_tuple=True)[0]
            if keep_idx.numel() > 0:
                wp = write_pos[keep_idx]
                for jj in range(4):
                    tokens_out[keep_idx, wp + jj] = sub_tokens[keep_idx, jj]
                    pos_out[keep_idx, wp + jj] = pos4[jj].unsqueeze(0)
                write_pos[keep_idx] += 4

            # ---- Images that MERGE (merge=True) ----
            merge_idx = (merge).nonzero(as_tuple=True)[0]
            if merge_idx.numel() > 0:
                wp = write_pos[merge_idx]
                tokens_out[merge_idx, wp] = merged[merge_idx]
                pos_out[merge_idx, wp] = self.merged_pos_embed[idx].unsqueeze(0)
                write_pos[merge_idx] += 1

        # Step 6: Build valid mask, add CLS token
        valid_mask = torch.zeros(B, K_max, dtype=torch.bool, device=x.device)
        for b in range(B):
            valid_mask[b, :K_per_image[b]] = True

        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_pos = self.pos_embed[:, 0:1, :].expand(B, -1, -1)
        h = torch.cat([cls_tokens, tokens_out], dim=1)
        pos = torch.cat([cls_pos, pos_out], dim=1)
        h = h + pos
        h = self.pos_drop(h)

        # Step 7: Transformer blocks — zero out padded positions
        pad_mask = torch.cat([
            torch.ones(B, 1, dtype=torch.bool, device=x.device),
            valid_mask,
        ], dim=1).unsqueeze(-1)  # (B, 1+K_max, 1)

        for block in self.blocks:
            h = block(h)
            h = h * pad_mask.float()

        h = self.norm(h)
        logits = self.head(h[:, 0])

        self._last_k = K_max
        self._last_n = self.num_patches
        self._last_merged = num_merged.float().mean().item()

        return logits


# ==============================================================================
# Training Utilities
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
    kept_tokens = []
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
            kept_tokens.append(model._last_k)
        pbar.set_postfix({
            'loss': f'{total_loss / (pbar.n if pbar.n else 1):.4f}',
            'acc': f'{100.0 * correct / total:.1f}%',
        })

    result = (total_loss / len(loader), 100.0 * correct / total)
    if track_patches and kept_tokens:
        avg_k = sum(kept_tokens) / len(kept_tokens)
        avg_n = model._last_n
        result = result + (avg_k, avg_n, avg_k / avg_n * 100)
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
    parser = argparse.ArgumentParser(description='Patch Merging-ViT Training')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=list(DATASETS.keys()))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accum_steps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--keep_ratio', type=float, default=0.75,
                        help='Ratio of blocks to keep (0.75 = keep 37/49 blocks)')
    parser.add_argument('--drop_path', type=float, default=0.1)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_only', type=str, default=None,
                        help='Path to checkpoint for evaluation only')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Dataset: {args.dataset}')
    print(f'Keep ratio: {args.keep_ratio} ({int(49 * args.keep_ratio)}/49 blocks)')
    print(f'Effective batch: {args.batch_size} × {args.accum_steps} = {args.batch_size * args.accum_steps}')

    # Dataset
    loader_fn, num_classes, default_epochs = DATASETS[args.dataset]
    if args.epochs is None:
        args.epochs = default_epochs
    result = loader_fn(batch_size=args.batch_size, data_dir='./data',
                       num_workers=args.num_workers)
    if len(result) == 4:
        train_loader, val_loader, test_loader, _ = result
    else:
        train_loader, test_loader, _ = result
        val_loader = test_loader

    # Save directory
    save_dir = f'./checkpoints/{args.dataset}_patch_merging_k{int(args.keep_ratio*100)}'
    os.makedirs(save_dir, exist_ok=True)

    # Model
    model = PatchMergingViT(
        num_classes=num_classes,
        keep_ratio=args.keep_ratio,
        pretrained=True,
        drop_path_rate=args.drop_path,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    router_params = sum(p.numel() for p in model.router.parameters()) / 1e6
    print(f'Total params: {total_params:.1f}M, Router params: {router_params:.3f}M')

    # Eval only
    if args.eval_only:
        ckpt = torch.load(args.eval_only, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        criterion = nn.CrossEntropyLoss()
        print(f'\n--- Evaluating {args.eval_only} ---')
        result = evaluate(model, test_loader, criterion, device,
                          track_patches=True, desc='Test')
        if len(result) == 5:
            test_loss, test_acc, avg_k, avg_n, keep_pct = result
            baseline = BASELINE_ACC.get(args.dataset, 0)
            print(f'Test Acc:        {test_acc:.2f}%')
            print(f'Baseline Acc:    {baseline:.2f}%')
            print(f'Acc Diff:        {test_acc - baseline:+.2f}%')
            print(f'Keep:            {avg_k:.0f}/{avg_n:.0f} tokens ({keep_pct:.1f}%)')
        else:
            test_loss, test_acc = result
            print(f'Test Acc: {test_acc:.2f}%')
        latency, throughput = compute_efficiency_metrics(model, test_loader, device)
        print(f'Latency:         {latency:.2f} ms/sample')
        print(f'Throughput:      {throughput:.2f} samples/sec')
        return

    # Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # Resume
    start_epoch = 0
    best_val_acc = 0.0
    best_epoch = -1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        best_epoch = ckpt.get('best_epoch', -1)
        print(f'Resumed from epoch {start_epoch}, best_acc={best_val_acc:.2f}%')
    else:
        # Auto-resume from latest checkpoint
        epoch_files = sorted(
            [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch_')],
            key=lambda f: int(f.split('_')[-1].split('.')[0]))
        if epoch_files:
            ckpt_path = os.path.join(save_dir, epoch_files[-1])
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_val_acc = ckpt.get('best_val_acc', 0.0)
            best_epoch = ckpt.get('best_epoch', -1)
            print(f'Auto-resumed from {ckpt_path}, epoch {start_epoch}')

    if start_epoch >= args.epochs:
        print(f'Training complete ({start_epoch}/{args.epochs}). Use --eval_only.')
        return

    # Save args
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Training loop
    print(f'\n--- Training {start_epoch} → {args.epochs} ---')
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            accum_steps=args.accum_steps, epoch=epoch+1, total_epochs=args.epochs)

        val_loss, val_acc, avg_k, avg_n, keep_pct = evaluate(
            model, val_loader, criterion, device, track_patches=True, desc='Val')
        scheduler.step()

        print(f'Epoch {epoch+1}/{args.epochs} | '
              f'Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | '
              f'Keep: {avg_k:.0f}/{avg_n:.0f} ({keep_pct:.1f}%) | '
              f'LR: {scheduler.get_last_lr()[0]:.2e}')

        # Save checkpoint
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
        }

        torch.save(checkpoint_data,
                   os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(checkpoint_data,
                       os.path.join(save_dir, 'best_model.pth'))
            print(f'  → New best: {val_acc:.2f}%')

    # Final evaluation
    print(f'\n--- Final Evaluation (best epoch {best_epoch}) ---')
    ckpt = torch.load(os.path.join(save_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    result = evaluate(model, test_loader, criterion, device,
                      track_patches=True, desc='Test')
    test_loss, test_acc, avg_k, avg_n, keep_pct = result

    baseline = BASELINE_ACC.get(args.dataset, 0)
    print(f'Test Acc:        {test_acc:.2f}%')
    print(f'Baseline Acc:    {baseline:.2f}%')
    print(f'Acc Diff:        {test_acc - baseline:+.2f}%')
    print(f'Keep:            {avg_k:.0f}/{avg_n:.0f} tokens ({keep_pct:.1f}%)')

    latency, throughput = compute_efficiency_metrics(model, test_loader, device)
    print(f'Latency:         {latency:.2f} ms/sample')
    print(f'Throughput:      {throughput:.2f} samples/sec')

    # Save final results
    final_results = {
        'test_acc': test_acc,
        'baseline_acc': baseline,
        'acc_diff': test_acc - baseline,
        'avg_tokens': avg_k,
        'avg_total': avg_n,
        'keep_pct': keep_pct,
        'latency_ms': latency,
        'throughput': throughput,
        'best_epoch': best_epoch,
        'args': vars(args),
    }
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f'\nResults saved to {save_dir}/results.json')


if __name__ == '__main__':
    main()
