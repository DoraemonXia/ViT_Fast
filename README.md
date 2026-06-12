# Miss Patch — Minimal Sufficient Visual Information for ViT Classification

当 Vision Transformer 处理图像时，不是所有 patch 都同样重要。本项目系统性地剥离信息维度（分辨率、颜色、频率、结构），探究 **图像分类到底需要多少信息**。

## 核心发现

**MAE + Router 可以在保留 75% patch 的情况下几乎不掉精度（-0.59%）**，并减少 25% 计算量。实验证明 ViT 的 patch 冗余度很高，通过学习型 Router 可以有效过滤不重要 patch。

| 方法 | CIFAR-100 | vs Baseline |
|:----|:--------:|:----------:|
| Baseline | 91.69% | — |
| MAE + Router (keep 75%) | **91.10%** | **-0.59%** |
| MAE + Router (keep 50%) | **89.07%** | **-2.62%** |
| APT Selection (keep 68%) | **85.84%** | **-5.85%** |
| APT Merge (keep 37%) | **83.13%** | **-8.56%** |

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
├── train_apt_patch_selection.py  # APT 熵值 patch selection（新）
├── train_apt_patch_merge.py      # APT 多尺度 patch merge（新）
├── eval_training_free_token_reduction.py # ToMe / EViT 即插即用评估
├── datasets.py                   # 数据集加载器
├── docs/
│   ├── research_story.md         # 研究叙事（给读者看）
│   ├── research_story.html       # HTML 版
│   ├── all_experiments_guide.png # 实验可视化对照图
│   └── resolution_sweep_guide.png# 分辨率扫描对照图
├── checkpoints/                  # 模型权重（见下方下载）
├── logs/                         # 训练日志
└── results/                      # 实验结果 CSV / JSONL
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

### 推理（本地权重）

```bash
# 环境
pip install torch torchvision timm pillow tqdm

# 运行全部模型（默认使用 checkpoints/ 下的本地权重）
python inference.py --all --dataset cifar100 --gpu 0

# 测试指定模型
python inference.py --model mae_router75 --dataset cifar100 --gpu 0
python inference.py --model mae_router50 --dataset cifar100 --gpu 0
```

| --model | 说明 | 默认数据集 |
|:--------|:-----|:---------:|
| `baseline` | ViT-B/16 全量 (196 patches) | cifar100 |
| `mae_router75` | MAE + Router 保留 75% (147 patches) | cifar100 |
| `mae_router50` | MAE + Router 保留 50% (98 patches) | cifar100 |
| `--all` | 运行全部模型 | — |

### Training-free token reduction（不训练）

脚本默认使用 CIFAR-100 的 full ViT-B/16 微调权重 `checkpoints/cifar100_vit_b16_ft` 作为 baseline，并优先从 `checkpoints/` 读取本地权重；只有显式加 `--download` 时才会联网下载。

```bash
# ToMe：每层逐步合并 r 个相似 token；保护 CLS，使用 proportional attention
python eval_training_free_token_reduction.py --method tome --dataset cifar100 --tome_r_values 4,8,13 --gpu 0

# EViT-style：用浅层 CLS→patch attention 直接 prune
python eval_training_free_token_reduction.py --method evit --dataset cifar100 --evit_keep_ratios 0.75,0.68,0.50 --evit_layer 3 --gpu 0

# 一次跑 baseline / ToMe / EViT
python eval_training_free_token_reduction.py --method all --dataset cifar100 --gpu 0
```

这组结果是 **training-free**，只适合和同样不训练的 ToMe / EViT / entropy heuristic 互相比较；APT Entropy、APT Merge、MAE+Router 属于已训练或微调方法，应放在单独表格里作为参考。

运行结果会自动追加到 `results/training_free_token_reduction.csv` 和 `results/training_free_token_reduction.jsonl`。如果只想在终端查看，不写文件，加 `--no_save_results`。

### 训练

```bash
# Baseline
python train.py --model vit_b16 --dataset cifar100 --epochs 100

# MAE + Router（需先蒸馏后训练）
python train_router_distill.py --dataset cifar100 --gpu 0
python train_patch_selection_mae.py --dataset cifar100 --gpu 0   --router_path ./checkpoints/router_distill_cifar100/router.pth

# APT Selection（熵值丢弃低信息量 patch）
python train_apt_patch_selection.py --dataset cifar100 --gpu 0 --threshold 5.5

# APT Merge（熵值低区域 2×2 合并为 1 token）
python train_apt_patch_merge.py --dataset cifar100 --gpu 0 --threshold 5.5
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
