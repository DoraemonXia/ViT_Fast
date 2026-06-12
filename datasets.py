import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image
import os


# Full replacement for ToTensor - use torch.tensor with numpy array conversion
def _to_tensor(pic):
    if isinstance(pic, np.ndarray):
        if pic.ndim == 2:
            pic = pic[:, :, None]
        # Use torch.tensor with explicit numpy array conversion
        img = torch.tensor(pic.copy().transpose(2, 0, 1), dtype=torch.float32) / 255.0
        return img.contiguous()

    # Handle PIL Image
    if not isinstance(pic, Image.Image):
        raise TypeError(f"pic should be PIL Image or ndarray. Got {type(pic)}")

    # Convert PIL to numpy then tensor
    img = np.array(pic, copy=True)
    if img.ndim == 2:
        img = img[:, :, None]
    img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32) / 255.0
    return img.contiguous()


class ToTensorFixed:
    def __call__(self, pic):
        return _to_tensor(pic)


def _split_indices(length, val_ratio, seed=42):
    n_val = int(length * val_ratio)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(length, generator=generator).tolist()
    return indices[n_val:], indices[:n_val]


def get_cifar10_loader(batch_size=64, data_dir='./data', num_workers=0, image_size=224):
    transform_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=image_size//8),
        transforms.RandomHorizontalFlip(),
        ToTensorFixed(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        ToTensorFixed(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, 10


def get_cifar100_loader(batch_size=64, data_dir='./data', num_workers=0,
                        image_size=224, use_randaugment=False,
                        return_val=False, val_ratio=0.1, split_seed=42):
    transform_list = [
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=image_size//8),
        transforms.RandomHorizontalFlip(),
    ]
    if use_randaugment:
        transform_list.append(transforms.RandAugment(num_ops=2, magnitude=9))
    transform_list += [
        ToTensorFixed(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]

    transform_train = transforms.Compose(transform_list)
    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        ToTensorFixed(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)

    if return_val:
        val_dataset_full = datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=transform_test)
        train_indices, val_indices = _split_indices(
            len(train_dataset), val_ratio, split_seed)
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset_full, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    if return_val:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
        return train_loader, val_loader, test_loader, 100
    return train_loader, test_loader, 100


def get_tiny_imagenet_loader(batch_size=64, data_dir='./data/tiny-imagenet', num_workers=0, image_size=224):
    """Tiny-ImageNet loader with optional resize to 224x224 for ViT."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=image_size // 8),
        transforms.RandomHorizontalFlip(),
        ToTensorFixed(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        ToTensorFixed(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, 200


def get_oxford_pets_loader(batch_size=32, data_dir='./data', num_workers=0, image_size=224, val_ratio=0.2):
    """Oxford-IIIT Pets loader with train/val split from the trainval set.

    Official split: trainval (3,680) + test (3,669), 37 classes.
    We further split trainval into train (80%) and val (20%).
    Used in ViT paper (Dosovitskiy et al., 2021).
    """
    from torchvision.datasets import OxfordIIITPet

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.RandomCrop(image_size, padding=image_size // 8),
        transforms.RandomHorizontalFlip(),
        ToTensorFixed(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        ToTensorFixed(),
        normalize,
    ])

    # Use separate dataset objects so validation never receives train augmentation.
    full_train_dataset = OxfordIIITPet(
        root=data_dir, split='trainval', download=True, transform=transform_train)
    full_val_dataset = OxfordIIITPet(
        root=data_dir, split='trainval', download=True, transform=transform_test)
    test_dataset = OxfordIIITPet(
        root=data_dir, split='test', download=True, transform=transform_test)

    train_indices, val_indices = _split_indices(len(full_train_dataset), val_ratio)
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, 37


def _make_generic_loader(dataset_class, batch_size, data_dir, num_workers, image_size,
                          val_ratio, train_split, test_split, num_classes, normalize=None):
    """Helper for datasets with train/test splits and optional val split from train."""
    if normalize is None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.RandomCrop(image_size, padding=image_size // 8),
        transforms.RandomHorizontalFlip(),
        ToTensorFixed(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        ToTensorFixed(),
        normalize,
    ])

    full_train = dataset_class(root=data_dir, split=train_split, download=True, transform=transform_train)
    full_val = dataset_class(root=data_dir, split=train_split, download=True, transform=transform_test)
    test_dataset = dataset_class(root=data_dir, split=test_split, download=True, transform=transform_test)

    train_indices, val_indices = _split_indices(len(full_train), val_ratio)
    train_dataset = Subset(full_train, train_indices)
    val_dataset = Subset(full_val, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, num_classes


def get_food101_loader(batch_size=32, data_dir='./data', num_workers=0, image_size=224, val_ratio=0.1):
    """Food-101: 101 classes, 75,750 train / 25,250 test."""
    from torchvision.datasets import Food101
    return _make_generic_loader(Food101, batch_size, data_dir, num_workers, image_size,
                                val_ratio, 'train', 'test', 101)


def get_stanford_cars_loader(batch_size=32, data_dir='./data', num_workers=0, image_size=224, val_ratio=0.1):
    """Stanford Cars: 196 classes, 8,144 images total."""
    from torchvision.datasets import StanfordCars
    return _make_generic_loader(StanfordCars, batch_size, data_dir, num_workers, image_size,
                                val_ratio, 'train', 'test', 196)


def get_dtd_loader(batch_size=32, data_dir='./data', num_workers=0, image_size=224, val_ratio=0.1):
    """DTD (Describable Textures): 47 classes, 5,640 images."""
    from torchvision.datasets import DTD
    return _make_generic_loader(DTD, batch_size, data_dir, num_workers, image_size,
                                val_ratio, 'train', 'test', 47)


def get_flowers102_loader(batch_size=32, data_dir='./data', num_workers=0, image_size=224, val_ratio=0.2):
    """Flowers-102: 102 classes, 1020 train / 1020 val / 6149 test.

    Official split: train (1020), val (1020), test (6149).
    We combine train+val, then split 80/20 for train/val.
    """
    from torchvision.datasets import Flowers102

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.RandomCrop(image_size, padding=image_size // 8),
        transforms.RandomHorizontalFlip(),
        ToTensorFixed(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        ToTensorFixed(),
        normalize,
    ])

    # Combine train + val splits for more training data
    train_part = Flowers102(root=data_dir, split='train', download=True, transform=transform_train)
    val_part = Flowers102(root=data_dir, split='val', download=True, transform=transform_train)
    train_part_eval = Flowers102(root=data_dir, split='train', download=True, transform=transform_test)
    val_part_eval = Flowers102(root=data_dir, split='val', download=True, transform=transform_test)
    test_dataset = Flowers102(root=data_dir, split='test', download=True, transform=transform_test)

    from torch.utils.data import ConcatDataset
    full_train = ConcatDataset([train_part, val_part])
    full_val = ConcatDataset([train_part_eval, val_part_eval])
    train_indices, val_indices = _split_indices(len(full_train), val_ratio)
    train_dataset = Subset(full_train, train_indices)
    val_dataset = Subset(full_val, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, 102
