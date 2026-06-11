# Miss Patch — Minimal Sufficient Visual Information for ViT Classification

当 Vision Transformer 处理图像时，不是所有 patch 都同样重要。本项目系统性地剥离信息维度（分辨率、颜色、频率、结构），探究 **图像分类到底需要多少信息**。

## 核心发现

**降采样 168×168 是最简单有效的方案** —— 只需一行 `Resize`，计算量减半，精度几乎不受影响（或根据数据集特性略有下降）。

| 方案 | 额外成本 | Oxford Pets | Food-101 |
|:----|:--------:|:-----------:|:--------:|
| 降采样 168×168 | 无 | -1.26% | -1.50% |
| 灰度图 | 无 | -1.23% | — |
| MAE + Router (50%) | 需训练 Router + Decoder | -3.11% | -1.85% |

**更详细的实验记录见 [EXPERIMENTS.md](EXPERIMENTS.md)**。研究故事见 [docs/research_story.md](docs/research_story.md)。

## 项目结构

```
├── models.py                     # 所有模型定义
│   ├── MAEPatchSelectionViT      # MAE + Router 主模型
│   ├── MAEDecoder                # MAE 解码器
│   ├── PatchSelectionViT         # Gumbel 方案（旧）
│   └── SemanticRouter            # MLP Router
├── train.py                      # 通用训练脚本（baseline / Gumbel）
├── train_patch_selection_mae.py  # MAE + Router 训练脚本
├── train_router_distill.py       # 注意力蒸馏脚本
├── test_downsample_train.py      # 降采样训练脚本
├── test_grayscale_train.py       # 灰度图训练脚本
├── test_preprocess_reduce.py     # 推理预处理测试
├── datasets.py                   # 数据集加载器
├── benchmark_fair.py             # 公平吞吐量基准
├── docs/
│   ├── research_story.md         # 研究叙事（给读者看）
│   ├── research_story.html       # HTML 版
│   ├── all_experiments_guide.png # 实验可视化对照图
│   └── resolution_sweep_guide.png# 分辨率扫描对照图
├── checkpoints/                  # 模型权重（见下方下载）
└── logs/                         # 训练日志
```

## 支持的数据集

| 数据集 | 类别 | 训练集 | 特点 | 下载方式 |
|:------|:---:|:-----:|:-----|:--------|
| CIFAR-100 | 100 | 50K | 32×32 原生，粗粒度 | `torchvision.datasets.CIFAR100` |
| Oxford Pets | 37 | 3.6K | 高分辨率，猫狗品种 | `torchvision.datasets.OxfordIIITPet` |
| Food-101 | 101 | 68K | 高分辨率，细粒度菜肴 | `torchvision.datasets.Food101` |
| DTD | 47 | 5.6K | 纹理分类 | `torchvision.datasets.DTD` |
| Flowers-102 | 102 | ~2K | 花卉品种 | `torchvision.datasets.Flowers102` |

所有数据集首次加载时会自动下载。

## 快速开始

### 推理（权重自动下载）

```bash
# 环境
pip install torch torchvision timm pillow

# 运行全部模型（自动从 Hugging Face 下载权重）
python inference.py --all --dataset cifar100 --gpu 0

# 测试指定模型
python inference.py --model downsample168 --dataset oxford_pets --gpu 0
python inference.py --model mae_router75 --dataset cifar100 --gpu 0
```

| --model | 说明 | 默认数据集 |
|:--------|:-----|:---------:|
| `baseline` | ViT-B/16 全量 (196 patches) | cifar100 |
| `downsample168` | 降采样 168x168 (100 patches) | cifar100 |
| `downsample112` | 降采样 112x112 (49 patches) | cifar100 |
| `mae_router75` | MAE + Router 保留 75% (147 patches) | cifar100 |
| `mae_router50` | MAE + Router 保留 50% (98 patches) | cifar100 |
| `--all` | 运行全部模型 | — |

### 训练

```bash
# Baseline
python train.py --model vit_b16 --dataset cifar100 --epochs 100

# MAE + Router（需先蒸馏后训练）
python train_router_distill.py --dataset cifar100 --gpu 0
python train_patch_selection_mae.py --dataset cifar100 --gpu 0   --router_path ./checkpoints/router_distill_cifar100/router.pth
```## 模型权重

预训练 ViT-B/16 来自 `timm`（`vit_base_patch16_224.augreg_in21k`），自动下载。

微调后的权重和 Router 权重详见 [EXPERIMENTS.md](EXPERIMENTS.md#检查点文件)。

## 引用

如果本项目对你的研究有帮助，请考虑引用：

```bibtex
@misc{misspatch2025,
  title = {Miss Patch: Minimal Sufficient Visual Information for ViT Classification},
  author = {ypxia},
  year = {2025}
}
```

## 许可

MIT
