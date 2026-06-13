# APT 实验结果分析

## 1. 实验概况

本文分析 Fixed 16/32 Patch Merge 与 Hierarchical 16/32 Learned APT（A4）的正式训练结果，以及 CIFAR-100 上最新完成的同环境速度测评。

主要结果文件：

```text
apt_experiments/results_downloaded/
├── CIFAR100/
│   ├── args.json
│   ├── history.json
│   ├── results.json
│   ├── cifar100_a4_speed.json
│   └── checkpoint_epoch_79.pth
└── OxfordPets/
    ├── args.json
    ├── history.json
    ├── results.json
    └── checkpoint_epoch_69.pth

checkpoints/cifar100_apt_merge_t5.5_s42/
├── args.json
├── history.json
├── results.json
├── best_model.pth
└── checkpoint_latest.pth
```

共同配置：

| 配置项 | 数值 |
|:---|:---|
| Backbone | ViT-B/16 IN-21K pretrained |
| 输入尺寸 | `224×224` |
| 原始 Patch Tokens | 196 |
| APT 层级 | 16×16 / 32×32 |
| 聚合方式 | Fixed Merge：平均聚合；A4：学习式加权聚合 |
| 熵直方图 | 64 bins |
| 训练 Batch / Accum | `32 / 4` |
| 学习率 | `3e-5` |
| 数值精度 | BF16 mixed precision |
| Seed | 42 |

CIFAR-100 Learned APT 使用阈值 `3.25`、训练 80 轮；Oxford Pets Learned APT 使用阈值 `4.0`、训练 70 轮。Fixed Merge 在 CIFAR-100 上使用阈值 `5.5`、训练 50 轮。

## 2. 最终精度结果

| 方法 | 数据集 | 轮数 | 最佳 Val Acc | 最佳轮次 | Test Acc | Baseline | 精度差 | 平均真实 Tokens | Token 减少 |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed 16/32 Merge | CIFAR-100 | 50 | 84.28% | 50 | **83.80%** | 91.69% | **-7.89%** | 49.76/196 | **74.61%** |
| Learned APT | CIFAR-100 | 80 | 90.90% | 76 | **91.07%** | 91.69% | **-0.62%** | 145.24/196 | **25.90%** |
| Learned APT | Oxford Pets | 70 | 95.24% | 44 | **93.00%** | 93.81% | **-0.81%** | 140.34/196 | **28.40%** |

Fixed Merge 实现了非常激进的压缩，但测试精度下降 `7.89` 个百分点，不满足“保持精度”的目标。Learned APT 在两个数据集上的测试精度损失均控制在 1 个百分点以内，实现了“减少约四分之一 Token，同时基本保持精度”。

## 3. Learned APT Token 分布

| 数据集 | Mean | Std | P50 | P90 | Min | Max | 真实保留率 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | 145.24 | 30.21 | 148 | 181 | 49 | 196 | 74.10% |
| Oxford Pets | 140.34 | 28.98 | 142 | 178 | 67 | 196 | 71.60% |

A4 会根据图像内容动态选择粒度，而不是采用固定裁剪比例。Oxford Pets 使用的 Token 更少，但两个数据集的 P90 仍接近完整的 196 Tokens，说明复杂图像会显著拉高同一 Batch 的最大序列长度。

## 4. Padding 与实际计算

| 方法/数据集 | 平均真实 Tokens | 平均 Padding 后 Tokens | 理论 Token 减少 | Padding 后序列减少 |
|:---|---:|---:|---:|---:|
| Fixed Merge / CIFAR-100 | 49.76 | 58.27 | 74.61% | **70.27%** |
| Learned APT / CIFAR-100 | 145.24 | 191.40 | 25.90% | **2.35%** |
| Learned APT / Oxford Pets | 140.34 | 190.78 | 28.40% | **2.66%** |

两种方法都需要把同一 Batch 内的样本补齐到该 Batch 的最长序列。Learned APT 的图像间 Token 差异较大，导致实际 Dense 序列仍接近 196；Fixed Merge 的绝大多数样本都被压缩到较短长度，Padding 后仍只需约 58 Tokens。

因此，Fixed Merge 更容易获得实际计算收益，但代价是严重的精度损失；Learned APT 保住了精度，却没有把真实 Token 减少充分转化为 Dense 序列缩短。

## 5. 速度对比

### 5.1 CIFAR-100 同环境实测

测试条件为 RTX 4090D、BF16、Batch Size 32、预热 20 Batch、每轮测量 100 Batch、重复 3 次并取中位数。

| 方法 | Tokens（真实/Padding） | Acc | 延迟 | 吞吐量 | 相对速度 | 峰值显存 |
|:---|---:|---:|---:|---:|---:|---:|
| Baseline ViT-B/16 | 196/196 | **91.72%** | **0.401 ms/sample** | **2494.4/s** | `1.000x` | 672.31 MiB |
| A4 Learned APT | 146.04/191.74 | **91.20%** | **0.450 ms/sample** | **2223.9/s** | **`0.892x`（-10.8%）** | 717.08 MiB |

本次同环境测评结论：

- A4 精度比 Baseline 低 `0.52` 个百分点。
- A4 真实 Tokens 减少 `25.49%`，但 Padding 后序列只减少 `2.17%`。
- A4 吞吐量下降 `10.84%`，单样本延迟增加 `12.16%`。
- A4 峰值显存增加约 `44.77 MiB`，即约 `6.66%`。

速度下降的主要原因是：接近完整长度的 Dense Padding 没有带来足够的 Transformer 计算节省，同时还增加了熵计算、动态 Token 构建、排序、Gather 和 Attention Mask 等额外开销。

`91.20%` 来自第 80 轮检查点的本次测速流程；第 2 节的 `91.07%` 来自训练完成后加载最佳验证检查点得到的正式 `results.json`。二者权重和评测时点不同，不属于结果冲突。

### 5.2 Fixed Merge 训练后测速

Fixed Merge 的 `results.json` 使用训练脚本内置测速流程，在 RTX 4090D、BF16、Batch Size 32 下得到：

| 方法 | Tokens（真实/Padding） | Test Acc | 延迟 | 吞吐量 | 峰值显存 |
|:---|---:|---:|---:|---:|---:|
| Fixed 16/32 Merge | 49.76/58.27 | **83.80%** | **0.251 ms/sample** | **3976.3/s** | 1699.21 MiB |

该结果说明激进且较稳定的短序列确实能够提高吞吐量。但此处的测速来自训练脚本，未使用第 5.1 节的独立 Benchmark 脚本进行三次重复，峰值显存还可能包含训练结束后未释放的优化器状态，因此不能与第 5.1 节直接计算严格加速比。

从研究目标看，Fixed Merge 的速度表现较好，但 `83.80%` 的测试精度明显不合格，不能作为最终方案。

### 5.3 与历史速度结果对照

| 方法 | Tokens | Acc | 吞吐量 | 加速 | 数据口径 |
|:---|---:|---:|---:|---:|:---|
| Baseline | 196 | 91.69% | 791/s | `1.00x` | 历史环境 |
| ToMe `r=8` | 100 | 91.34% | 819/s | `+4%` | 历史环境 |
| MAE+Router 75% | 147 | 91.10% | 1004/s | `+27%` | 历史环境 |
| MAE+Router 50% | 98 | 89.07% | 1368/s | `+73%` | 历史环境 |
| ToMe `r=13` | 40 | 90.74% | 1122/s | `+42%` | 历史环境 |
| Fixed Merge 50 epochs | 49.76/58.27 | 83.80% | 3976.3/s | 不直接计算 | 本次训练脚本测速 |
| **A4 Learned APT** | 146.04/191.74 | 91.20% | 2223.9/s | **-10.8%** | 本次 RTX 4090D |

历史吞吐量与 A4 本次数据来自不同硬件和评测流程，绝对数值不能直接横向比较。可比较的是各方法在各自同环境 Baseline 下的相对变化：

- MAE+Router 75% 与 A4 的真实 Token 数接近，精度也接近，但历史 MAE+Router 获得了 `+27%` 加速。
- MAE+Router 使用固定长度 Top-K，Batch 内无需补齐到接近 196，因此更容易转化为实际加速。
- Fixed Merge 将 Padding 后长度压缩到约 58，速度收益明显，但精度下降 `7.89` 个百分点。
- A4 当前虽然能为每张图像动态分配 Token，但变长序列经过 Padding 后丢失了主要计算优势。

## 6. 收敛情况

### Fixed Merge / CIFAR-100

- 50 轮训练记录完整，最佳验证精度出现在第 50 轮，为 `84.28%`。
- 第 7 轮历史验证精度为 `83.13%`，继续训练到第 50 轮只提高 `1.15` 个百分点。
- 最终测试精度为 `83.80%`，相对 Baseline 下降 `7.89` 个百分点。
- 第 50 轮训练精度达到 `99.98%`，但验证精度仍只有 `84.28%`，存在严重过拟合。
- 这证明旧 Merge 表现差并不主要是因为只训练了 7 轮，而是固定平均合并和过高压缩率本身造成了信息损失。

### CIFAR-100

- 最佳验证精度出现在第 76 轮，为 90.90%。
- 第 80 轮验证精度为 90.88%，已经进入稳定平台。
- 后期训练精度接近 100%，继续增加轮数的收益有限。

### Oxford Pets

- 最佳验证精度出现在第 44 轮，为 95.24%。
- 第 70 轮验证精度为 94.84%，比最佳值低约 0.41 个百分点。
- 后期训练精度达到 100%，存在明显过拟合。

## 7. 与历史策略对比

历史实验的 Val/Test 口径并不完全一致，表格主要用于判断策略演进趋势。

| 方法 | 数据集 | Tokens | 精度 | vs Baseline | 说明 |
|:---|:---|---:|---:|---:|:---|
| Entropy Selection | CIFAR-100 | 133 | 85.84% Test | -5.85% | 直接删除低熵 Patch |
| Fixed 16/32 Merge | CIFAR-100 | 49.76 | 83.80% Test | -7.89% | 完整 50 轮正式结果 |
| MAE+Router 50% | CIFAR-100 | 98 | 89.07% Test | -2.62% | 固定保留 50% |
| MAE+Router 75% | CIFAR-100 | 147 | 91.10% Test | -0.59% | 固定保留 75% |
| **Learned APT** | CIFAR-100 | 145.24 | **91.07% Test** | **-0.62%** | 动态层级合并 |
| **Learned APT** | Oxford Pets | 140.34 | **93.00% Test** | **-0.81%** | 动态层级合并 |

从精度看，Learned APT 明显优于 Entropy Selection 和完整训练 50 轮的 Fixed Merge，并与 MAE+Router 75% 接近。CIFAR-100 上 Learned APT 的测试精度比 Fixed Merge 高 `7.27` 个百分点。Learned APT 的优势是按图像动态分配 Token 并保持精度；Fixed Merge 的优势是压缩率高、Padding 后序列短；MAE+Router 75% 则在精度和固定长度加速之间取得了更好的平衡。

## 8. 实验结论

1. Learned APT 在 CIFAR-100 和 Oxford Pets 上分别减少约 `25.9%` 和 `28.4%` 的真实 Tokens。
2. 两个数据集的正式测试精度损失均小于 1 个百分点，说明动态层级合并策略在精度层面可行。
3. Fixed Merge 完整训练 50 轮后测试精度仍只有 `83.80%`；相较第 7 轮，最佳验证精度仅提高 `1.15` 个百分点，说明增加训练轮数无法解决其根本问题。
4. Fixed Merge 将真实 Tokens 减少 `74.61%`，Padding 后序列也减少 `70.27%`，具备实际加速条件，但精度损失过大。
5. Learned APT 整体设计将 CIFAR-100 测试精度从 Fixed Merge 的 `83.80%` 提高到 `91.07%`。该提升同时受到学习式聚合、更保守的阈值和层级编码影响，不能只归因于单一组件。
6. 当前 Learned APT 实现没有实现“保持精度的同时减少实际计算资源消耗”：CIFAR-100 同环境实测吞吐下降 `10.8%`，延迟增加 `12.2%`，峰值显存增加 `6.7%`。
7. Learned APT 的问题不在训练轮数或模型精度，而在推理实现：真实 Tokens 虽然减少，但 Batch Padding 使实际序列仍达到约 `191.7/196`，额外动态处理开销超过了节省的计算。
8. 因此，Fixed Merge 实现了加速潜力但不能保持精度；Learned APT 实现了精度保持但当前不能加速，二者都尚未同时满足项目目标。

> 最终结论：50 轮 Fixed Merge 证明激进合并可以缩短实际计算序列，但会造成不可接受的精度损失；Learned Hierarchical APT 能将精度损失控制在 1% 内，却受 Batch Padding 影响无法获得实际加速。当前两个方案分别满足了项目目标的一半。
