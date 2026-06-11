"""
Unified inference: evaluate all model variants.
Auto-downloads weights from Hugging Face if not found locally.

Usage:
  python inference.py --model baseline --dataset cifar100
  python inference.py --model downsample168 --dataset oxford_pets
  python inference.py --model mae_router75 --dataset cifar100
  python inference.py --all --dataset oxford_pets
"""
import torch, argparse, os, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import get_cifar100_loader, get_oxford_pets_loader, get_food101_loader
from models import create_model
import timm

HF_BASE = 'https://hf-mirror.com/1999xia/ViT_Fast/resolve/main'

DATASETS = {
    'cifar100':    (get_cifar100_loader, 100, 224, 'Suitable for: baseline, mae_router'),
    'oxford_pets': (get_oxford_pets_loader, 37, 224, 'Suitable for: all models (recommended)'),
    'food101':     (get_food101_loader, 101, 224, 'Suitable for: baseline, mae_router'),
}

# Model registry: (name, description, patches, default_dataset, checkpoint_subpath, model_creator)
MODELS = [
    ('baseline', 'ViT-B/16 full (224x224, 196 patches)', 196, 'cifar100',
     'cifar100_vit_b16_ft',
     lambda nc: timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True, num_classes=nc)),

    ('downsample168', 'Downsampled 168x168 (100 patches)', 100, 'oxford_pets',
     'oxford_pets_vit_b16_img168',
     lambda nc: timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True, num_classes=nc, img_size=168)),

    ('downsample112', 'Downsampled 112x112 (49 patches)', 49, 'oxford_pets',
     'oxford_pets_vit_b16_img112',
     lambda nc: timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True, num_classes=nc, img_size=112)),

    ('mae_router75', 'MAE + Router keep 75% (147 patches)', 147, 'cifar100',
     'cifar100_mae_patchsel_b16_keep75_distill',
     lambda nc: create_model('mae_patch_selection_vit_b16', num_classes=nc, keep_ratio=0.75,
                             pretrained=True, decoder_embed_dim=512, decoder_depth=4)),

    ('mae_router50', 'MAE + Router keep 50% (98 patches)', 98, 'cifar100',
     'cifar100_mae_patchsel_b16_keep50',
     lambda nc: create_model('mae_patch_selection_vit_b16', num_classes=nc, keep_ratio=0.5,
                             pretrained=True, decoder_embed_dim=512, decoder_depth=4)),
]


def find_checkpoint(subpath):
    """Try multiple checkpoint location patterns."""
    patterns = [
        f'checkpoints/{subpath}/best_model.pth',
        f'checkpoints/{subpath}/router.pth',
        f'checkpoints/{subpath}',
        f'checkpoints/{subpath}.pth',
    ]
    for p in patterns:
        if os.path.exists(p):
            return p
    # Return preferred path for download (will try both dir and flat)
    return f'checkpoints/{subpath}/best_model.pth'


def download(url, target_path):
    """Download checkpoint from HF. Tries both dir and flat structure."""
    if os.path.exists(target_path):
        return True

    # Determine the subpath name from target_path
    subname = target_path.split('/')[-2] if target_path.endswith('.pth') else target_path.split('/')[-1]

    # Try all possible download URLs
    alt_paths = [
        target_path,  # dir/best_model.pth
        target_path.replace(f'/{subname}/best_model.pth', f'/{subname}'),  # flat name
        target_path.replace(f'/{subname}/router.pth', f'/{subname}'),  # flat name (router)
    ]
    alt_paths = list(dict.fromkeys(alt_paths))  # deduplicate

    for path in alt_paths:
        if os.path.exists(path):
            return True
        os.makedirs(os.path.dirname(path), exist_ok=True)
        hf_url = f'{HF_BASE}/{path}'
        print(f'  Downloading: {hf_url}')
        for attempt in range(2):
            try:
                urllib.request.urlretrieve(hf_url, path)
                print(f'  Saved to: {path}')
                return True
            except Exception:
                if attempt == 0:
                    continue
    print(f'  Failed to download {subname}')
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
    parser.add_argument('--model', type=str, default='all',
                        choices=[m[0] for m in MODELS] + ['all'])
    parser.add_argument('--dataset', type=str, default=None,
                        choices=list(DATASETS.keys()))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}\n', flush=True)

    models_to_run = MODELS if args.model == 'all' else [m for m in MODELS if m[0] == args.model]

    for name, desc, patches, default_ds, ckpt_subpath, create_fn in models_to_run:
        dataset = args.dataset or default_ds
        ds_info = DATASETS[dataset]
        print(f'--- {desc} ---')
        print(f'Dataset: {dataset} ({ds_info[2]})')

        loader_fn, num_classes, img_size, _ = ds_info
        _, _, test_loader, _ = loader_fn(batch_size=args.batch_size, data_dir='./data', num_workers=4)

        # Find / download checkpoint
        ckpt_path = find_checkpoint(ckpt_subpath)
        hf_url = f'{HF_BASE}/checkpoints/{ckpt_subpath}'

        if not os.path.exists(ckpt_path):
            download(hf_url, ckpt_path)
            # Also try alternate path if first download fails
            alt_path = f'checkpoints/{ckpt_subpath}/best_model.pth'
            if not os.path.exists(ckpt_path) and os.path.exists(alt_path):
                ckpt_path = alt_path

        if not os.path.exists(ckpt_path):
            print(f'  No checkpoint found at {ckpt_path}')
            print()
            continue

        # Load model
        model = create_fn(num_classes)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        sd = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(sd, strict=False)

        # Evaluate
        acc = evaluate(model, test_loader, device)
        print(f'  Test Acc: {acc:.2f}%  (Baseline: varies by dataset)')
        print(flush=True)
        del model
        torch.cuda.empty_cache()

    print('Done!')

if __name__ == '__main__':
    main()
