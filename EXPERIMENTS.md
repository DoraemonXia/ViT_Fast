# Miss Patch — Pre-Tokenization Patch Selection for Vision Transformers

在 Vision Transformer（ViT）前插入一个 Router/Filter 模块，动态过滤不重要的 image patches，减少计算量，同时尽量保持分类精度。

---

## 项目文件

| 文件 | 说明 |
|------|------|
| `models.py` | 所有模型定义，见下方"模型类"表格 |
| `train.py` | 通用训练脚本（支持 baseline、patch selection、MAE finetune 等） |
| `train_patch_selection_mae.py` | MAE patch selection 专用训练脚本（含蒸馏 router 加载） |
| `train_router_distill.py` | Attention Distillation 训练脚本 |
| `train_apt_patch_selection.py` | **APT 熵值 patch selection 训练脚本（新）** |
| `train_apt_patch_merge.py` | **APT 多尺度 patch merge 训练脚本（新）** |
| `test_patch_selection_b16.py` | 旧 Gumbel patch selection 测试脚本 |
| `test_blur_downsample.py` | 图片模糊/降采样测试脚本 |
| `test_stride_patches.py` | 不同 stride 下 patch 数量 vs 精度测试脚本 |
| `datasets.py` | 数据集加载器：CIFAR-10/100, Oxford Pets, Food-101, Tiny-ImageNet, DTD, Flowers-102, Stanford Cars |
| `checkpoints/` | 所有模型权重 |
| `logs/` | 训练日志 |
| `SHARED_MEMORY.md` | 跨 Claude Code 会话共享记忆 |

### models.py — 模型类

| 类名 | 说明 |
|------|------|
| `GumbelSelection` | Gumbel-Softmax 可微分选择模块（含 STE + 退火温度） |
| `SemanticRouter` | 语义 Router（MLP: D→D/2→1 + GELU + LayerNorm） |
| `PatchSelectionViT` | Patch Selection ViT（Router + 可微分 Top-K + Gumbel 噪声） |
| `RandomPruneViT` | 随机丢弃 patch 的 baseline |
| `MAEDecoder` | MAE 解码器（4 blocks, 512-dim transformer） |
| `MAEViT` | 标准 MAE（mask 75% patches + 重建） |
| `MAEPatchSelectionViT` | **MAE + Patch Selection**（Router + Differentiable Top-K + 轻量 encoder + 10 blocks backbone + MAE Decoder） |

### models.py — create_model() 可用 model_name

| model_name | 对应模型 | 说明 |
|-----------|---------|------|
| `swin_tiny` | Swin-Tiny | Swin Transformer baseline |
| `patch_selection_vit` | PatchSelectionViT | Gumbel 方案（旧） |
| `patch_selection_vit_b16` | PatchSelectionViT | ViT-B/16 IN-21K + Gumbel |
| `patch_selection_vit_b16_in1k` | PatchSelectionViT | ViT-B/16 IN-1K + Gumbel |
| `random_prune_vit` | RandomPruneViT | 随机丢弃 50% baseline |
| `mae_vit` | MAEViT | 标准 MAE 预训练 |
| `mae_patch_selection_vit_b16` | MAEPatchSelectionViT | **MAE + 蒸馏 Router（推荐）** |

---

## 使用方法

### 1. 训练 Baseline（全量 ViT-B/16）

```bash
# CIFAR-100 (IN-21K pretrained)
python train.py --model vit_b16 --dataset cifar100 --lr 3e-5 --epochs 100 --batch_size 128

# Oxford Pets
python train.py --model vit_b16 --dataset oxford_pets --lr 3e-5 --epochs 100 --batch_size 32

# Food-101
python train.py --model vit_b16 --dataset food101 --lr 3e-5 --epochs 30 --batch_size 32
```

### 2. 训练 MAE Patch Selection（+ 蒸馏 Router）

**两步走：**

**Step 1:** 训练蒸馏 Router（从教师 ViT 学习注意力分布）
```bash
python train_router_distill.py --dataset cifar100 --gpu 4
# 可选参数：
#   --batch_size 64     # 批量大小（默认 64）
#   --lr 1e-4           # 学习率（默认 1e-4）
#   --epochs 5          # 训练轮数（默认 5）
#   --weight_decay 0.05 # 权重衰减

# 输出：checkpoints/router_distill_{dataset}/router.pth
```

**Step 2:** 用蒸馏 Router 初始化，训练完整模型
```bash
python train_patch_selection_mae.py --dataset cifar100 --gpu 4 \
  --router_path ./checkpoints/router_distill_cifar100/router.pth

# 可选参数：
#   --keep_ratio 0.5    # 保留 patch 比例（默认 0.5，推荐 0.75）
#   --batch_size 32     # 批量大小（默认 32）
#   --accum 4           # 梯度累积步数（默认 4，有效 batch = 128）
#   --lr 3e-5           # 学习率（默认 3e-5）
#   --weight_decay 0.05 # 权重衰减
#   --label_smoothing 0.1
#   --mse_start 1.0     # MSE 损失起始权重
#   --mse_end 0.1       # MSE 损失结束权重（cosine anneal）
#   --decoder_dim 512   # MAE decoder 维度
#   --decoder_depth 4   # MAE decoder 层数
#   --router_path PATH  # 蒸馏 router 权重路径（可选）
```

**不加 `--router_path` 就是随机初始化 Router**（原始 MAE 方案）。

### 3. 训练 Gumbel Patch Selection（旧方案，不推荐）

```bash
python train.py --model patch_selection_vit_b16 --dataset cifar100
```

### 4. 测试图片预处理（模糊/降采样）

```bash
python test_blur_downsample.py --dataset cifar100 --gpu 5
# 测试 Gaussian blur、降采样等预处理对精度的影响（不减少 token 数量）
```

### 5. 测试 Stride-based Patch 减少

```bash
python test_stride_patches.py --dataset cifar100 --gpu 5
# 测试不同 stride 下 patch 数量 vs 精度（直接减少 token 数量）
```

### 6. 降采样图片训练（减小输入分辨率）

```bash
# 112×112 → 49 patches (75% reduction)
python test_downsample_train.py --dataset cifar100 --image_size 112 --gpu 3

# 168×168 → 100 patches (49% reduction)
python test_downsample_train.py --dataset cifar100 --image_size 168 --gpu 5

# 可选参数：
#   --batch_size 128   # 批量大小（默认 128）
#   --lr 3e-5          # 学习率（默认 3e-5）
#   --epochs 100       # 训练轮数（默认使用数据集默认值）
#   --label_smoothing 0.1
#   --weight_decay 0.05
#   --num_workers 4

# 输出：checkpoints/{dataset}_vit_b16_img{size}/best_model.pth
```

### 7. 通用训练脚本 train.py 全部参数

```bash
python train.py \
  --model vit_b16                # 模型名
  --dataset cifar100             # 数据集
  --batch_size 128               # 批量大小
  --epochs 100                   # 训练轮数
  --lr 3e-5                      # 学习率
  --weight_decay 0.05            # 权重衰减
  --keep_ratio 0.5               # patch 保留比例
  --selection_mode topk          # 选择模式
  --adaptive_alpha 0.5           # 自适应 alpha
  --patch_size 16                # patch 大小
  --patch_stride 16              # patch stride
  --num_workers 10               # 数据加载线程
  --use_randaugment              # 使用 RandAugment
  --label_smoothing 0.0          # 标签平滑
  --device auto                  # 设备
  --save_dir ./checkpoints       # 保存目录
  --pretrained                   # 使用预训练权重
  --load_mae PATH                # 加载 MAE 预训练权重
  --log_dir ./logs               # 日志目录
  --drop_path 0.0                # DropPath 率
  --mixup 0.0                    # MixUp alpha
  --cutmix 0.0                   # CutMix alpha
  --ema_decay 0.0                # EMA 衰减率
  --clip_grad 1.0                # 梯度裁剪
  --warmup_epochs 0              # 预热轮数
```

### 支持的数据集

`datasets.py` 提供：
- `get_cifar10_loader` — CIFAR-10 (10 classes, 50K train)
- `get_cifar100_loader` — CIFAR-100 (100 classes, 50K train)
- `get_oxford_pets_loader` — Oxford Pets (37 classes, ~3.6K train)
- `get_food101_loader` — Food-101 (101 classes, 68K train)
- `get_tiny_imagenet_loader` — Tiny-ImageNet (200 classes, 100K train)
- `get_dtd_loader` — DTD (47 classes, texture)
- `get_flowers102_loader` — Flowers-102 (102 classes)
- `get_stanford_cars_loader` — Stanford Cars (196 classes)

---

## 项目架构

### Baseline 架构（全量 ViT-B/16 IN-21K）

```
Image (224×224) → Patch Embed (196 patches, 768-dim) → 12 ViT-B blocks → CLS → Classifier
```

### MAE Patch Selection 架构（当前最佳）

```
Image → Patch Embed + Pos Embed → Router (MLP 768→384→1) → Sigmoid STE Top-K (keep 75%)
  → 选中 patches (147个) + CLS → 轻量 Encoder (前 2 个 ViT-B block)
    → 主干 (后 10 个 ViT-B block) → CLS → CE Loss
    → MAE Decoder (4 blocks, 512-dim) → 重建丢弃 patches → MSE Loss (λ: 1.0→0.1)
```

### Attention Distillation 流程

```
教师 ViT-B/16（冻结）→ 提取最后 block 的 CLS→patch 注意力 → 注意力分数 (B, 196)
MLP Router（学生）→ 从 patch embedding 预测重要性分数 → MSE Loss
→ 训练好的 Router 权重→ 初始化 MAEPatchSelectionViT → 端到端微调
```

---

## 实验结果汇总

### 主数据集：Oxford-IIIT Pets（37 类，高分辨率 ~200-1000px）

**全量对比（相同条件：batch=16, accum=2, lr=3e-5, 100 epochs）：**

| 方法 | 保留 | Test Acc | vs Baseline |
|:----|:----:|:--------:|:----------:|
| **Baseline（全量 ViT-B/16）** | 100% | **91.91%** | — |
| 降采样 168x168 | 51% patches | **90.65%** | **-1.26%** |
| 灰度图（Grayscale） | 亮度 only | **90.68%** | **-1.23%** |
| **MAE + 从头训练 Router** | 50% patches | **88.80%** | **-3.11%** |
| 降采样 112x112 | 25% patches | **86.64%** | **-5.27%** |
| **MAE + 预热 Router** | 50% patches | **85.85%** | **-6.06%** |

**结论：**
- 降采样 168x168 和灰度图效果最好（掉 ~1.2%），且不需要额外模块
- MAE + 从头训练 Router（88.80%）比预热过的 Router（85.85%）更好——在小数据集上，教师蒸馏信号太弱（重叠率仅 51.16%≈随机），预热反而有害
- 降采样比所有学习型方案更简单、效果更好

---

### 验证数据集

#### Food-101（101 类，细粒度菜肴，~300-512px）

| 方法 | Test Acc | vs Baseline |
|:----|:--------:|:----------:|
| Baseline（batch=128） | 91.37% | — |
| 降采样 168x168 | 89.87% | -1.50% |
| 降采样 112x112 | **85.96%** | -5.41% |
| MAE + 预热 Router（50%，CIFAR） | 89.52% | -1.85%（参考） |

#### DTD（47 类，纹理分类，~300-400px）

| 方法 | Test Acc | vs Baseline |
|:----|:--------:|:----------:|
| Baseline | 80.85% | — |
| 降采样 168x168 | **74.20%** | **-6.65%** |

---

### 跨数据集对比：降采样 168x168

| 数据集 | 特点 | Baseline | 168x168 | 下降幅度 |
|:------|:-----|:-------:|:-------:|:--------:|
| CIFAR-100 | 粗粒度，原生 32x32 | 91.69% | 91.56% | -0.13%（假象） |
| Food-101 | 细粒度菜肴 | 91.37% | 89.87% | -1.50% |
| **Oxford Pets** | 猫狗品种，花纹纹理 | **~93.3%** | **90.65%** | **-3.16%** |
| DTD | 纯纹理分类 | 80.85% | 74.20% | **-6.65%** |

**规律：分类粒度越细、越依赖高频纹理，降采样伤害越大。**

---

### 旧实验（CIFAR-100，仅供参考）

> CIFAR-100 原生仅 32x32，降采样结果不可靠，但 Router 类实验（不改变图片大小）有效。

| 方法 | 保留 | CIFAR-100 | vs BL |
|:----|:----:|:--------:|:-----:|
| Baseline | 100% | 91.69% | — |
| Gumbel Selection | 50% | 87.18% | -4.51% |
| MAE + 随机 Router | 50% | 88.25% | -3.44% |
| MAE + 蒸馏 Router | 50% | 89.07% | -2.62% |
| MAE + 蒸馏 Router | 75% | 91.10% | -0.59% |## 关键发现

1. **Gumbel 方案最差** — Gumbel 噪声导致 Router 梯度几乎消失，无法有效学习 patch 重要性
2. **MAE 重建损失有帮助** — 让 Router 通过"能否重建丢弃 patch"来学习，比纯分类信号好（+1%）
3. **Attention Distillation 最佳** — 用教师 ViT 的 CLS 注意力预训练 Router，再端到端微调
   - CIFAR-100: 88.25% → **89.07%**（+0.82%）
   - Food-101: ~87.97% → **89.52%**（+1.55%）
4. **数据集大小重要** — Oxford Pets 仅 ~3.6K 训练图像，蒸馏信号太弱，反而下降
5. **CLS token flow 关键 bug** — 早期实现中 CLS 只走 10/12 block（前 2 block 浪费），修复后 epoch 1 从 80.63%→84.69%
6. **75% keep ratio 是更优平衡点** — 只掉 0.59%（vs 50% 掉 2.62%），推荐使用
7. **降低 token 质量 vs 减少 token 数量** — 模糊/降采样图片几乎不影响精度（-0.14~-0.57%），但计算量完全不变；减少 token 数量才真正降低计算量
8. **Naive stride 减少 token 效果差且实用性低** — stride=32 减少 75% token 时掉 21%，远不如学习型 Router 或降采样训练

---

## APT-style 熵值 Patch Selection 实验

将 APT (Adaptive Patch Tokenization, ICLR 2026) 的熵值评估方法与 ViT_Fast 框架结合，验证基于 handcrafted 熵特征的 patch 选择策略效果。

### 方法

用 APT 的熵值计算替代可学习 Router：对每张图计算 16×16 patch 的像素熵，保留熵值高于阈值的 patch（自适应每张图保留量），丢弃低熵区域。

### 训练

```bash
# CIFAR-100
python train_apt_patch_selection.py --dataset cifar100 --gpu 0 --threshold 5.5

# Oxford Pets
python train_apt_patch_selection.py --dataset oxford_pets --gpu 0 --threshold 5.5

# Food-101
python train_apt_patch_selection.py --dataset food101 --gpu 0 --threshold 5.5

# 断点续训（自动从最新 checkpoint 恢复）
# 直接重新运行相同命令即可

# 仅评估（不训练）
python train_apt_patch_selection.py --dataset cifar100 --gpu 0 --eval_only ./checkpoints/cifar100_apt_entropy_t5.5/best_model.pth

# 自定义阈值
python train_apt_patch_selection.py --dataset cifar100 --gpu 0 --threshold 4.0 --min_keep 16 --max_keep_ratio 0.8
```

### 参数

| 参数 | 说明 | 默认值 |
|:-----|:-----|:------:|
| `--threshold` | 熵值阈值（越高保留越少） | 5.5 |
| `--min_keep` | 每张图至少保留的 patch 数 | 32 |
| `--max_keep_ratio` | 最大保留比例 | 0.9 |
| `--multi_scale` | 启用多尺度（16+32）合并 | False |
| `--resume PATH` | 从指定 checkpoint 恢复 | — |
| `--eval_only PATH` | 仅评估 checkpoint | — |

### 结果

*训练配置：50 epochs, lr=3e-5, batch=32×4=128, label_smoothing=0.1, GPU: RTX 5070 Ti*

| 数据集 | 方法 | Threshold | Keep% | Test Acc | vs Baseline (91.69%) |
|:------|:-----|:--------:|:-----:|:--------:|:----------:|
| CIFAR-100 | APT Entropy | 5.5 | 68.3% (133/196) | **85.84%** | **-5.85%** |
| CIFAR-100 | MAE+Router (ref) | — | 50% | 89.07% | -2.62% |
| CIFAR-100 | MAE+Router (ref) | — | 75% | **91.10%** | **-0.59%** |

**发现：** 熵值选择在 CIFAR-100 上效果不及可学习 Router。CIFAR-100 原生 32×32，放大到 224 后，16×16 patch 熵值区分度有限——低熵区域（平坦背景）和高熵区域（物体边缘）的熵差不够大，导致选择策略误杀有用 patch。

### 与可学习 Router 对比

| 方法 | Acc (CIFAR-100) | Keep% | 优势 | 劣势 |
|:-----|:------:|:-----:|:-----|:-----|
| APT Entropy (本实验) | 85.84% | 68.3% | 无需训练，开箱即用，0 额外参数 | 手工特征，CIFAR 上区分度差 |
| MAE + 蒸馏 Router | **89.07%** | 50% | 端到端学习，注意力引导 | 需要教师 + 蒸馏训练 |
| MAE + 随机 Router | 88.25% | 50% | 端到端学习 | 收敛慢 |

---

## APT-style 多尺度 Patch Merge 实验

实现与原始 APT 论文一致的多尺度合并策略：对熵值低的 2×2 patch 块（32×32 区域），将 4 个 16×16 sub-patches pooling 为 1 个 token，而不是简单丢弃。

### 方法

1. 计算 32×32 patch 的熵值
2. 熵值 < threshold → 合并 4 个 sub-patches 为 1 token（平均 pooling）
3. 熵值 >= threshold → 保持 4 个独立 tokens
4. 合并 token 使用 7×7 位置编码（从 14×14 重采样）

### 训练

```bash
# CIFAR-100
python train_apt_patch_merge.py --dataset cifar100 --gpu 0 --threshold 5.5

# Oxford Pets
python train_apt_patch_merge.py --dataset oxford_pets --gpu 0 --threshold 5.5

# 仅评估
python train_apt_patch_merge.py --dataset cifar100 --gpu 0 --eval_only PATH
```

### 参数

| 参数 | 说明 | 默认值 |
|:-----|:-----|:------:|
| `--threshold` | 32×32 熵值阈值（低于此值合并） | 5.5 |
| `--resume PATH` | 从指定 checkpoint 恢复 | — |
| `--eval_only PATH` | 仅评估 checkpoint | — |

### 与 selection 策略对比

| 策略 | 低熵区域处理 | 保留信息 | 实现 |
|:-----|:-----------|:-------|:-----|
| **APT Selection** | 丢弃 patch | 丢失 | 简单 |
| **APT Merge** (本实验) | 4→1 合并 | 保留（粗粒度） | 接近原始 APT |

### 结果

*merge 在训（当前 epoch 7/50），以下为最新结果。训练配置：50 epochs, lr=3e-5, GPU: RTX 5070 Ti*

| 数据集 | Threshold | 原始 tokens | 合并后 tokens | Val Acc | vs Baseline (91.69%) |
|:------|:--------:|:----------:|:------------:|:--------:|:----------:|
| CIFAR-100 | 5.5 | 196 | **73 (37.2%)** | **83.13%** | **-8.56%** |
| CIFAR-100 | MAE+Router 50% (ref) | 196 | 98 (50%) | 89.07% | -2.62% |
| CIFAR-100 | APT Selection (ref) | 196 | 133 (68.3%) | 85.84% | -5.85% |

**发现：** 合并策略 token 数量最少（37.2%），但精度下降也最大（-8.56%）。CIFAR-100 上 32×32 熵值阈值 5.5 过于激进——大量 2×2 块被合并，丢失了放大后的细节。需降低阈值或在更高分辨率数据集上测试。merge 策略在 token 压缩率上有明显优势（37.2% vs Selection 68.3%），但当前精度换不来压缩率。

### 两种 APT 策略对比（CIFAR-100, threshold=5.5）

| 策略 | Tokens | Val Acc | 优势 | 适用场景 |
|:-----|:------:|:-------:|:-----|:-----|
| **Selection** | 133 (68.3%) | **85.84%** | 精度更好，实现简单 | 对精度要求高 |
| **Merge** | 73 (37.2%) | 83.13% | Token 数少 45%，计算更小 | 对速度要求高，或高分辨率原图 |
| **MAE+Router** | 98 (50%) | 89.07% | 最佳精度/压缩平衡 | 有教师模型可用时 |

---

```
checkpoints/
├── cifar100_mae_patchsel_b16_keep50/
│   └── best_model.pth          —— 89.07% (MAE + 蒸馏 Router, keep 50%)
├── cifar100_mae_patchsel_b16_keep75/
│   └── best_model.pth          —— 91.10% (MAE + 蒸馏 Router, keep 75%) ★推荐
├── food101_mae_patchsel_b16_keep50/
│   └── best_model.pth          —— 89.52% (MAE + 蒸馏 Router, keep 50%)
├── oxford_pets_mae_patchsel_b16_keep50/
│   └── best_model.pth          —— 85.99% (MAE + 蒸馏 Router, keep 50%)
├── cifar100_patchsel_b16_keep50/
│   └── best_model.pth          —— 87.18% (Gumbel, CIFAR-100)
├── food101_patchsel_b16_keep50/
│   └── best_model.pth          —— 85.77% (Gumbel, Food-101)
├── oxford_pets_patchsel_b16_keep50/
│   └── best_model.pth          —— 87.00% (Gumbel, Oxford Pets)
├── cifar100_random_prune_vit_keep50_topk/
│   └── best_model.pth          —— 48.13% (随机丢弃 50%)
├── cifar100_vit_b16_ft/
│   └── best_model.pth          —— 88.75% (全量微调, CIFAR-100)
├── cifar100_vit_small/
│   ├── best_model.pth          —— 84.83% (ViT-S/16 baseline)
│   └── checkpoint_epoch*.pth   —— 每 10 epoch 检查点
├── cifar100_patch_selection_vit_keep50_topk/
│   ├── best_model.pth          —— 73.01% (旧 Gumbel 实验)
│   └── checkpoint_epoch*.pth   —— 每 10 epoch 检查点
├── vit_b16_cifar10_best.pth    —— CIFAR-10 baseline
├── vit_b16_cifar10_ft/
│   └── best_model.pth          —— CIFAR-10 微调
├── dtd_vit_b16_in21k/
│   └── best_model.pth          —— DTD 80.85%
├── flowers102_vit_b16_in21k/
│   └── best_model.pth          —— Flowers-102 100.00%
├── food101_vit_b16_in21k/
│   └── best_model.pth          —— Food-101 86.68% (val)
├── oxford_pets_vit_b16_ft/
│   └── best_model.pth          —— Oxford Pets 全量微调
├── oxford_pets_vit_b16_in21k/
│   └── best_model.pth          —— Oxford Pets IN-21K 95.65% (val)
├── mae_cifar100_mask75/
│   ├── mae_best.pth            —— MAE 预训练最佳
│   ├── mae_encoder_final.pth   —— MAE encoder 最终
│   ├── mae_epoch50.pth         —— 50 epoch 检查点
│   └── mae_epoch100.pth        —— 100 epoch 检查点
├── router_distill_cifar100/
│   └── router.pth              —— 蒸馏 Router (CIFAR-100, 55.76% overlap)
├── router_distill_food101/
│   └── router.pth              —— 蒸馏 Router (Food-101, 59.95% overlap)
└── router_distill_oxford_pets/
    └── router.pth              —— 蒸馏 Router (Oxford Pets, 51.16% overlap)
```

---

## 超参数（推荐）

所有 MAE patch selection 实验统一使用：

| 参数 | 值 |
|:----|:---|
| Backbone | ViT-B/16 IN-21K pretrained (`vit_base_patch16_224.augreg_in21k`) |
| Batch size | 32 |
| Gradient accumulation | 4（effective batch = 128） |
| Learning rate | 3e-5 |
| Weight decay | 0.05 |
| Label smoothing | 0.1 |
| Scheduler | CosineAnnealingLR |
| Gradient clip | 1.0 |
| Keep ratio | **0.75**（147/196 patches，推荐）或 0.5（98/196 patches） |
| MSE weight | 1.0 → 0.1（cosine anneal） |
| Decoder dim | 512 |
| Decoder depth | 4 |
| Epochs | CIFAR-100: 100, Oxford Pets: 100, Food-101: 30 |

---

*最后更新: 2026-05-04*
