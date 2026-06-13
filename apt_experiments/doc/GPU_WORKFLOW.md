# APT 当前方案

APT 实验现只保留两种层次化方案。

## A4：主实验

**Hierarchical 16/32 Learned APT**

- 入口：`apt_experiments/Hierarchical_16_32_Learned_APT/train.py`
- 高熵区域保留 `16x16` token。
- 低熵 `32x32` 区域合并为一个 token。
- 使用轻量可学习加权聚合，并保留 scale encoding。
- GPU 队列只运行 A4。

## A3：备用方案

**Hierarchical 16/32 Average APT**

- 入口：`apt_experiments/Hierarchical_16_32_Average_APT/train.py`
- 区域划分与 A4 相同，但粗 token 使用平均池化。
- 不进入默认 GPU 队列。
- 只有 A4 训练异常或学习聚合没有收益时才运行。

## 已删除方案

- Entropy Selection；
- Fixed 16/32 Merge 训练入口；
- 无 scale encoding 消融；
- 16/32/64 扩展。

原 Fixed Merge 的历史结果仅保留为最终对照：

| 数据集 | Threshold | 原始 tokens | 合并后 tokens | Val Acc | vs Baseline |
|:-------|----------:|------------:|--------------:|--------:|------------:|
| CIFAR-100 | 5.5 | 196 | 73（37.2%） | 83.13% | -8.56% |

该结果来自约第 7 epoch，属于 preliminary reference，不是完整训练结果。

GPU 操作见 [`GPU训练流程及耗时估计.md`](GPU训练流程及耗时估计.md)。


