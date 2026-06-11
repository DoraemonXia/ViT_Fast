---
license: mit
library_name: timm
---

# ViT_Fast — Model Weights

This repository contains model checkpoints for the **Miss Patch** project.

**Code and documentation**: [github.com/DoraemonXia/ViT_Fast](https://github.com/DoraemonXia/ViT_Fast)

## Available Weights

### Baselines (ViT-B/16, IN-21K pretrained)
| File | Dataset | Test Acc |
|:-----|:-------|:--------:|
| `checkpoints/cifar100_vit_b16_ft/best_model.pth` | CIFAR-100 | 91.69% |
| `checkpoints/oxford_pets_vit_b16_ft/best_model.pth` | Oxford Pets | 93.32% |
| `checkpoints/food101_vit_b16_in21k/best_model.pth` | Food-101 | 91.37% |

### Downsampled (168x168)
| File | Dataset | Test Acc |
|:-----|:-------|:--------:|
| `checkpoints/cifar100_vit_b16_img168/best_model.pth` | CIFAR-100 | 91.56% |
| `checkpoints/oxford_pets_vit_b16_img168/best_model.pth` | Oxford Pets | 90.65% |

### MAE + Router
| File | Dataset | Keep Ratio | Test Acc |
|:-----|:-------|:---------:|:--------:|
| `checkpoints/cifar100_mae_patchsel_b16_keep75_distill/best_model.pth` | CIFAR-100 | 75% | 91.10% |
| `checkpoints/cifar100_mae_patchsel_b16_keep50/best_model.pth` | CIFAR-100 | 50% | 89.07% |

### Distilled Routers
| File | Dataset |
|:-----|:-------|
| `checkpoints/router_distill_cifar100/router.pth` | CIFAR-100 |
| `checkpoints/router_distill_food101/router.pth` | Food-101 |

