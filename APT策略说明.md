# APT 策略说明

## 1. 研究目标

本项目研究的问题是：Vision Transformer 处理图像时，不同 patch 的信息量并不相同，能否在尽量保持分类精度的前提下，减少送入 Transformer 的 token 数量和实际计算开销。

APT 路线目前依次验证了三种方案：

1. Entropy Patch Selection：直接删除低熵 patch。
2. Fixed 16/32 Patch Merge：将低熵区域的 4 个细 token 平均合并为 1 个粗 token。
3. Hierarchical 16/32 Learned APT：保留层次化合并，但使用可学习聚合器生成粗 token。

以下结果均以 CIFAR-100 的 Full ViT-B/16 baseline `91.69%` 为参照。需要注意，历史 Selection 记录的是 Test Acc，历史 Merge 和当前 A4 主要记录 Val Acc，因此现阶段横向比较用于判断方案趋势，最终报告仍应统一使用相同的数据划分和测试指标。

---

## 2. 方法 A：Entropy Patch Selection

### 2.1 策略

对输入图像的每个 `16x16` patch 计算灰度信息熵：

- 熵值高于阈值：认为包含较多纹理或边缘信息，保留该 patch。
- 熵值低于阈值：认为信息量较低，直接删除该 patch。
- 保留数量过少时，用熵值 Top-K 保证每张图至少保留一定数量的 patch。
- 对保留数量设置上限，避免大部分图像仍使用完整 token 序列。

该方案不引入可学习 Router，patch 决策完全由手工熵特征完成。

### 2.2 执行方法

历史实现文件：

```text
train_apt_patch_selection.py
```

对应执行形式：

```bash
python train_apt_patch_selection.py \
  --dataset cifar100 \
  --gpu 0 \
  --threshold 5.5 \
  --min_keep 32 \
  --max_keep_ratio 0.9 \
  --batch_size 32 \
  --accum 4 \
  --epochs 50
```

### 2.3 主要参数

| 参数 | 数值 |
|:-----|:-----|
| Backbone | ViT-B/16 IN-21K pretrained |
| 输入尺寸 | `224x224` |
| 基础 patch | `16x16`，共 196 tokens |
| 熵阈值 | `5.5` |
| 最少保留 | 32 tokens |
| 最大保留比例 | 90% |
| 训练轮数 | 50 epochs |
| Batch / Accum | `32 / 4`，有效 batch 128 |
| 学习率 | `3e-5` |
| Weight decay | `0.05` |
| Label smoothing | `0.1` |

### 2.4 实验结果

| 数据集 | 方法 | 训练进度 | 平均 tokens | Token 保留率 | Token 减少率 | Test Acc | vs Baseline |
|:-------|:-----|:---------|:------------:|:------------:|:------------:|:--------:|:-----------:|
| CIFAR-100 | Entropy Patch Selection | 50 epochs | 133/196 | 68.3% | 31.7% | 85.84% | -5.85% |

### 2.5 发现的问题

- 低熵不等于无用。背景、轮廓和大面积颜色区域仍可能为分类提供信息，直接删除后无法恢复。
- 熵是与分类目标无关的手工统计特征，容易误删对类别判断有用的 patch。
- CIFAR-100 原图只有 `32x32`，放大到 `224x224` 后，patch 熵值会受到插值影响，区分度有限。
- 虽然减少了约 31.7% token，但精度下降 5.85%，精度与压缩率之间的平衡较差。

---

## 3. 方法 B：Fixed 16/32 Patch Merge

### 3.1 策略

将 `14x14` 个 `16x16` patch 划分为 `7x7` 个 `32x32` 区域，并计算每个 `32x32` 区域的熵：

- 熵值低于阈值：把区域内 4 个 `16x16` token 平均池化为 1 个粗 token。
- 熵值高于或等于阈值：保留原来的 4 个细 token。
- 合并 token 使用对应区域的位置编码，然后和未合并 token 一起输入 ViT。

相比 Selection，该方案不再直接丢弃整个低熵区域，而是用一个粗 token 保留区域信息。

### 3.2 执行方法

历史实现文件：

```text
train_apt_patch_merge.py
```

对应执行形式：

```bash
python train_apt_patch_merge.py \
  --dataset cifar100 \
  --gpu 0 \
  --threshold 5.5 \
  --batch_size 32 \
  --accum 4 \
  --epochs 50 \
  --seed 42 \
  --pretrained_checkpoint apt_experiments/pretrained/model.safetensors \
  --num_workers 8 \
  --log_interval 20
```

当前脚本已优化为 BF16 混合精度和批量 token 构建，但固定平均合并策略本身没有改变。训练只滚动保存一个 `checkpoint_latest.pth`，同时保留 `best_model.pth`，避免再次因逐轮 checkpoint 占满数据盘。

### 3.3 主要参数

| 参数 | 数值 |
|:-----|:-----|
| Backbone | ViT-B/16 IN-21K pretrained |
| 输入尺寸 | `224x224` |
| 细粒度 patch | `16x16` |
| 粗粒度区域 | `32x32` |
| 聚合方式 | 4 个 token 平均池化为 1 个 token |
| 熵阈值 | `5.5` |
| 计划训练轮数 | 50 epochs |
| 实际记录进度 | 约第 7 轮 |
| Batch / Accum | `32 / 4`，有效 batch 128 |
| 学习率 | `3e-5` |
| Weight decay | `0.05` |
| Label smoothing | `0.1` |
| 数值精度 | BF16 mixed precision |
| Checkpoint | 仅保留 latest 与 best |

### 3.4 已记录结果

该实验没有完成计划的 50 轮，下面是约第 7 轮时保存的阶段结果，不能视为完整训练后的最终成绩。

| 数据集 | 方法 | 训练进度 | 平均 tokens | Token 保留率 | Token 减少率 | Val Acc | vs Baseline |
|:-------|:-----|:---------|:------------:|:------------:|:------------:|:-------:|:-----------:|
| CIFAR-100 | Fixed 16/32 Patch Merge | 约 7/50 epochs | 73/196 | 37.2% | 62.8% | 83.13% | -8.56% |

### 3.5 发现的问题

- 阈值 `5.5` 过于激进，大量 `32x32` 区域被压缩，平均只保留 73 个 token。
- 简单平均池化会抹平 4 个子 patch 之间的差异，重要局部信息和边缘细节被弱化。
- Token 数虽然最低，但精度损失也最大，当前压缩率无法换来可接受的准确率。
- 实验只运行到早期轮次，训练不充分；但结果已经表明“高阈值 + 固定平均合并”的配置风险较高。

---

## 4. 方法 C：Hierarchical 16/32 Learned APT

### 4.1 策略

当前优化方案保留方法 B 的层次化 `16x16/32x32` 结构，但重新设计粗 token 的生成方式：

- 高熵 `32x32` 区域：保留 4 个 `16x16` 细 token。
- 低熵 `32x32` 区域：仍压缩为 1 个粗 token，但不再直接求平均。
- 使用轻量 learned aggregator 为 4 个子 token 学习权重，再加权生成粗 token。
- 使用 scale encoding 区分细 token 和粗 token。
- 使用 masked attention 屏蔽 batch 内动态序列补齐产生的 padding token。

优化后的实现还将逐图片 Python token 构建改成批量 GPU 张量操作，并启用 BF16、TF32 和 fused AdamW。RTX 4090 上训练速度从约 `1.2 batch/s` 提升至约 `17 batch/s`。

### 4.2 执行方法

当前主实验文件：

```text
apt_experiments/Hierarchical_16_32_Learned_APT/train.py
```

当前 CIFAR-100 正式训练命令：

```bash
python -u apt_experiments/Hierarchical_16_32_Learned_APT/train.py \
  --dataset cifar100 \
  --gpu 0 \
  --batch_size 32 \
  --accum 4 \
  --epochs 80 \
  --seed 42 \
  --entropy_bins 64 \
  --threshold32 3.25 \
  --pretrained_checkpoint apt_experiments/pretrained/model.safetensors \
  --num_workers 8 \
  --log_interval 20 \
  2>&1 | tee -a apt_experiments/a4_cifar100.log
```

### 4.3 主要参数

| 参数 | 数值 |
|:-----|:-----|
| Backbone | ViT-B/16 IN-21K pretrained |
| 输入尺寸 | `224x224` |
| 细/粗粒度 | `16x16 / 32x32` |
| 粗 token 聚合 | Learned weighted aggregation |
| 熵直方图 bins | 64 |
| `32x32` 熵阈值 | `3.25` |
| 计划训练轮数 | 80 epochs |
| Seed | 42 |
| Batch / Accum | `32 / 4`，有效 batch 128 |
| 学习率 | `3e-5` |
| Weight decay | `0.05` |
| Label smoothing | `0.1` |
| 数值精度 | BF16 mixed precision |
| GPU | RTX 4090D |

### 4.4 约第 40–49 轮阶段结果

目前可确认的 checkpoint 为 `checkpoint_epoch_48.pth`，即已经完成 49 个 epoch。该 checkpoint 可以正常读取：

| 数据集 | 方法 | 训练进度 | 平均真实 tokens | Token 保留率 | Token 减少率 | 当前 Val Acc | Best Val Acc | vs Baseline | 训练 Acc |
|:-------|:-----|:---------|:----------------:|:------------:|:------------:|:------------:|:------------:|:-----------:|:--------:|
| CIFAR-100 | Hierarchical 16/32 Learned APT | 49/80 epochs | 约 145.2/196 | 约 74.1% | 约 25.9% | 90.18% | 90.80% | -0.89% | 后期约 98% |

该结果仍是中期结果，80 轮训练结束后的 `Test Acc`、延迟、吞吐量和峰值显存尚未生成。

### 4.5 发现的问题

- 当前最佳验证精度仍比 Full ViT baseline 低约 0.89%，尚未完全做到零精度损失。
- 训练准确率约 98%，验证准确率约 90%，存在一定过拟合迹象；继续增加轮数未必持续提高精度。
- 平均真实 token 为 145.2，但同一个 batch 需要补齐到最长序列，日志中 padded token 经常达到 `175–196`，因此实际计算节省可能低于理论 token 减少率。
- 原脚本每个 epoch 保存约 985MB 的完整 checkpoint，训练到第 50 轮时填满 50GB 数据盘，并导致 `checkpoint_epoch_49.pth` 损坏。后续应只保留最近 checkpoint、最佳模型和定期里程碑 checkpoint。
- 最终能否证明“减少计算资源”不能只看真实 token 数，还必须比较 Full ViT 的延迟、吞吐量和峰值显存。

---

## 5. 三种方案阶段对比

| 方法 | 低熵区域处理 | Tokens | 保留率 | 当前精度 | vs Baseline | 结果状态 |
|:-----|:-------------|:------:|:------:|:--------:|:-----------:|:---------|
| Entropy Selection | 直接删除 | 133/196 | 68.3% | 85.84% Test | -5.85% | 50 轮历史结果 |
| Fixed Merge | 4→1 平均合并 | 73/196 | 37.2% | 83.13% Val | -8.56% | 约第 7 轮阶段结果 |
| Learned Hierarchical APT | 4→1 可学习聚合 | 145.2/196 | 74.1% | 90.80% Best Val | -0.89% | 约第 49 轮阶段结果 |

从当前结果可以得到以下阶段性结论：

1. Selection 证明了单纯依靠熵值删除 patch 会造成明显的信息损失。
2. Fixed Merge 虽然保留了低熵区域，但过高的压缩率和固定平均聚合进一步放大了精度损失。
3. Learned Hierarchical APT 将精度差距从 `-5.85%/-8.56%` 缩小到约 `-0.89%`，说明“保留区域信息 + 可学习聚合”的改进方向有效。
4. 当前 A4 用约 25.9% 的真实 token 减少换取不足 1% 的最佳验证精度下降，明显优于两个历史 APT 方案。
5. A4 是否最终成立，仍需等待 80 轮训练完成，并在相同测试集上统一比较 Test Acc、延迟、吞吐量和峰值显存。

## 6. 当前判断

从约第 40–49 轮的训练结果来看，Hierarchical 16/32 Learned APT 已经表现出比 Selection 和 Fixed Merge 更好的精度/压缩平衡，说明当前优化方向具有可行性。

不过，现阶段更准确的表述应是：

> A4 已初步验证“在平均减少约四分之一真实 token 的同时，将 CIFAR-100 最佳验证精度损失控制在 1% 内”；最终结论等待 80 轮训练、独立测试集评估和真实性能测速完成后确定。
