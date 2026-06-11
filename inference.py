"""
Unified inference: evaluate all model variants on any dataset.
Auto-downloads weights from Hugging Face if not found locally.

Usage:
  python inference.py --model baseline --dataset cifar100
  python inference.py --model downsample168 --dataset oxford_pets
  python inference.py --model mae_router75 --dataset cifar100
  python inference.py --all --dataset cifar100   # run all models
"""
import torch
import torch.nn as nn
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import get_cifar100_loader, get_oxford_pets_loader, get_food101_loader
from models import create_model
import timm

HF_BASE = 'https://hf-mirror.com/1999xia/ViT_Fast/resolve/main'

DATASET_INFO = {
    'cifar100':    (get_cifar100_loader, 100),
    'oxford_pets': (get_oxford_pets_loader, 37),
    'food101':     (get_food101_loader, 101),
}

MODEL_REGISTRY = {
    'baseline': {
        'desc': 'ViT-B/16 full (224×224, 196 patches)',
        'checkpoint': 'checkpoints/cifar100_vit_b16_ft/best_model.pth',
        'hf_path': f'{HF_BASE}/checkpoints/cifar100_vit_b16_ft/best_model.pth',
        'create': lambda nc: timm.create_model(
            'vit_base_patch16_224.augreg_in21k', pretrained=True, num_classes=nc),
        'img_size': 224,
    },
    'downsample168': {
        'desc': 'Downsampled 168×168 (100 patches)',
        'checkpoint': 'checkpoints/cifar100_vit_b16_img168/best_model.pth',
        'hf_path': f'{HF_BASE}/checkpoints/cifar100_vit_b16_img168/best_model.pth',
        'create': lambda nc: timm.create_model(
            'vit_base_patch16_224.augreg_in21k', pretrained=True, num_classes=nc, img_size=168),
        'img_size': 168,
    },
    'downsample112': {
        'desc': 'Downsampled 112×112 (49 patches)',
        'checkpoint': 'checkpoints/cifar100_vit_b16_img112/best_model.pth',
        'hf_path': f'{HF_BASE}/checkpoints/cifar100_vit_b16_img112/best_model.pth',
        'create': lambda nc: timm.create_model(
            'vit_base_patch16_224.augreg_in21k', pretrained=True, num_classes=nc, img_size=112),
        'img_size': 112,
    },
    'grayscale': {
        'desc': 'Grayscale (Oxford Pets)',
        'checkpoint': 'checkpoints/oxford_pets_vit_b16_grayscale/best_model.pth',
        'hf_path': None,
        'create': lambda nc: timm.create_model(
            'vit_base_patch16_224.augreg_in21k', pretrained=True, num_classes=nc),
        'img_size': 224,
    },
    'mae_router75': {
        'desc': 'MAE + Router (keep 75%, 147 patches)',
        'checkpoint': 'checkpoints/cifar100_mae_patchsel_b16_keep75_distill/best_model.pth',
        'hf_path': f'{HF_BASE}/checkpoints/cifar100_mae_patchsel_b16_keep75_distill/best_model.pth',
        'create': lambda nc: create_model(
            'mae_patch_selection_vit_b16', num_classes=nc, keep_ratio=0.75,
            pretrained=True, decoder_embed_dim=512, decoder_depth=4),
        'img_size': 224,
    },
    'mae_router50': {
        'desc': 'MAE + Router (keep 50%, 98 patches)',
        'checkpoint': 'checkpoints/cifar100_mae_patchsel_b16_keep50/best_model.pth',
        'hf_path': f'{HF_BASE}/checkpoints/cifar100_mae_patchsel_b16_keep50/best_model.pth',
        'create': lambda nc: create_model(
            'mae_patch_selection_vit_b16', num_classes=nc, keep_ratio=0.5,
            pretrained=True, decoder_embed_dim=512, decoder_depth=4),
        'img_size': 224,
    },
}


def download_checkpoint(url, target_path, max_retries=3):
    """Download checkpoint from Hugging Face if not exists."""
    if os.path.exists(target_path):
        print(f'  Checkpoint exists: {target_path}')
        return True
    if url is None:
        print(f'  No download URL for {target_path}')
        return False
    import urllib.request
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    print(f'  Downloading from HF: {url}')
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, target_path)
            print(f'  Saved to: {target_path}')
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f'  Retry {attempt+1}/{max_retries}: {e}')
            else:
                print(f'  Download failed: {e}')
                return False
    return False


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval().to(device)
    correct = total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        _, pred = logits.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=list(MODEL_REGISTRY.keys()) + ['all'], default='all')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=list(DATASET_INFO.keys()))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}', flush=True)

    loader_fn, num_classes = DATASET_INFO[args.dataset]
    _, _, test_loader, _ = loader_fn(batch_size=args.batch_size, data_dir='./data', num_workers=4)
    print(f'Dataset: {args.dataset}, Test: {len(test_loader.dataset)}, Classes: {num_classes}', flush=True)

    models_to_test = list(MODEL_REGISTRY.items()) if args.model == 'all' else [(args.model, MODEL_REGISTRY[args.model])]
    # Filter to models that match this dataset (rough name match)
    models_to_test = [(k, v) for k, v in models_to_test if args.dataset in k or k in ['baseline', 'downsample168', 'downsample112', 'mae_router75', 'mae_router50']]

    print(f'\n{"Model":<40} {"Patches":>8} {"Acc":>8}')
    print('-' * 60, flush=True)

    for name, cfg in models_to_test:
        checkpoint_path = cfg['checkpoint'].replace('cifar100', args.dataset) if 'cifar100' in cfg['checkpoint'] else cfg['checkpoint']
        hf_path = cfg['hf_path'].replace('cifar100', args.dataset) if cfg['hf_path'] and 'cifar100' in cfg['hf_path'] else cfg['hf_path']

        # Try with dataset-specific checkpoint
        if not os.path.exists(checkpoint_path) and hf_path:
            download_checkpoint(hf_path, checkpoint_path)

        if not os.path.exists(checkpoint_path) and os.path.exists(cfg['checkpoint']):
            checkpoint_path = cfg['checkpoint']

        patches = {'baseline': 196, 'downsample168': 100, 'downsample112': 49,
                   'grayscale': 196, 'mae_router75': 147, 'mae_router50': 98}.get(name, '?')

        if not os.path.exists(checkpoint_path):
            print(f'{cfg["desc"]:<40} {patches:>8} {"No checkpoint":>8}')
            continue

        model = cfg['create'](num_classes)
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        sd = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(sd, strict=False)
        acc = evaluate(model, test_loader, device)
        print(f'{cfg["desc"]:<40} {patches:>8} {acc:>7.2f}%', flush=True)
        del model
        torch.cuda.empty_cache()

    print(f'\nDone!', flush=True)


if __name__ == '__main__':
    main()
