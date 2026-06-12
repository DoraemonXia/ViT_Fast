# APT GPU 实验流程与候选方案

本文档只描述 `apt_experiments/` 独立子工程。所有命令均从仓库根目录
`ViT_Fast/` 执行。

## 1. 项目目录

```text
ViT_Fast/
├── data/                              # 与原项目共用的数据集
├── datasets.py                        # 与原项目共用的数据加载器
└── apt_experiments/
    ├── apt_utils.py                    # entropy、mask、token 统计
    ├── train_apt_patch_selection.py    # APT Selection
    ├── train_apt_patch_merge.py        # 固定 16/32 Merge
    ├── train_hierarchical_apt.py       # 层次化 APT
    ├── requirements.txt
    ├── ROADMAP.md
    ├── GPU_WORKFLOW.md                 # 本文档
    ├── scripts/
    │   ├── scan_apt_thresholds.py
    │   ├── generate_gpu_experiments.py
    │   └── aggregate_experiment_results.py
    ├── experiments/
    │   ├── scans/                      # CPU 阈值扫描结果
    │   ├── queues/                     # GPU 命令队列
    │   └── results/                    # 汇总结果
    └── checkpoints/                    # 后续 GPU 训练自动创建
```

APT 的 checkpoint 不再写入原项目的 `checkpoints/`，而是写入
`apt_experiments/checkpoints/`。

## 2. 最终验收目标

APT 配置只有同时满足以下两项才算有效：

1. 相对同协议 Full ViT，测试准确率下降不超过 `1.0` 个百分点。
2. 在同一 GPU、batch size 和 AMP 设置下，包含 entropy 与 token 组装开销后，
   端到端延迟、吞吐量或峰值显存至少改善 `10%`。

平均真实 token 减少至少 `20%` 只是进入正式训练的筛选条件，不是最终成功证据。

## 3. 当前 APT 候选方案

### A1：Entropy Selection

- 文件：`apt_experiments/train_apt_patch_selection.py`
- 对每个 `16x16` patch 计算 Shannon entropy。
- 高于阈值的 patch 保留，低于阈值的 patch 丢弃。
- 优点：最简单，token 压缩直接。
- 风险：低熵区域的信息完全丢失。
- 状态：进入第一轮 GPU 筛选。

### A2：Fixed 16/32 Average Merge

- 文件：`apt_experiments/train_apt_patch_merge.py`
- 将每个 `2x2` 细 patch 视为一个 `32x32` 区域。
- 低熵区域由 4 个 token 平均合并为 1 个；高熵区域保留 4 个 token。
- 优点：低熵区域仍保留粗粒度信息。
- 风险：平均池化可能损失结构，固定两层粒度不够灵活。
- 状态：进入第一轮 GPU 筛选。

### A3：Hierarchical 16/32 Average APT

- 文件：`apt_experiments/train_hierarchical_apt.py`
- 从粗区域向下递归判断是否细分，叶节点覆盖整张图且不重叠。
- 支持 `16x16`、`32x32` 两级 token。
- 粗 token 使用平均聚合，并加入可学习 scale encoding。
- 优点：实现了真正的层次化区域划分。
- 状态：进入第一轮 GPU 筛选。

### A4：Hierarchical 16/32 Learned APT

- 基于 A3，将平均聚合替换为轻量可学习加权聚合。
- 命令参数：`--aggregation learned`。
- 优点：可能比固定平均池化更好地保留判别信息。
- 风险：增加参数和计算开销。
- 状态：仅当 A3 接近精度门槛时进入第二轮。

### A5：Hierarchical APT Without Scale Encoding

- 基于 A3，关闭 patch 尺度编码。
- 命令参数：`--no_scale_encoding`。
- 用途：确认 scale encoding 是否真正有效。
- 状态：消融方案，不参与第一轮；只有 A3 成功时才运行。

### A6：Hierarchical 16/32/64 APT

- 输入分辨率改为 `256x256`，增加 `64x64` 粗 token。
- 命令参数：`--image_size 256 --enable_64 --threshold64 VALUE`。
- 聚合可选 `average` 或 `learned`。
- 风险：分辨率和 baseline 同时改变，不能直接与 `224x224` 结果混合比较。
- 状态：可选扩展，不阻塞核心结论。

### 明确排除的伪候选

`train_apt_patch_selection.py --multi_scale` 当前会回退到普通 Selection，并没有形成真正
的多尺度 token 序列，因此不得作为独立方案参与筛选。

## 4. 筛选空间

### 第一轮：固定最小矩阵

| 方案 | CIFAR-100 | Oxford Pets | token 目标 | epoch | seed |
|:-----|:---------:|:-----------:|:----------:|:-----:|:----:|
| A1 Selection | 是 | 是 | 约 75% | 5 | 42 |
| A2 Fixed Merge | 是 | 是 | 约 75% | 5 | 42 |
| A3 Hierarchical Average | 是 | 是 | 约 75% | 5 | 42 |

共 6 个任务。第一轮只使用 validation accuracy 与预扫描 token 比例筛选，不使用 test
结果选择方法。

### 第二轮：仅对第一轮胜出者

1. 增加约 50% token 预算点。
2. 若 A3 胜出或接近胜出，追加 A4 Learned Aggregation。
3. A5 只作为必要消融。
4. 每个数据集最多保留 1 至 2 个方法进入正式训练。

### 当前候选阈值

| 数据集 | Selection 75% / 50% | Merge/Hierarchical 75% / 50% |
|:-------|:--------------------:|:--------------------------------:|
| CIFAR-100 | `2.00` / `3.00` | `3.25` / `4.25` |
| Oxford Pets | `2.75` / `3.75` | `4.00` / `4.75` |
| Food-101 | `3.00` / `3.75` | `4.00` / `4.75` |
| DTD | `3.00` / `4.00` | `4.00` / `4.75` |

Food-101 和 DTD 已完成 CPU 扫描，但默认不进入第一轮 GPU 队列。

## 5. GPU 环境准备

推荐使用带 CUDA PyTorch 的云镜像，GPU 为 RTX 4090 24 GB 或同等级设备。

进入仓库根目录：

```bash
cd /path/to/ViT_Fast
```

安装 APT 独立依赖：

```bash
python -m pip install -r apt_experiments/requirements.txt
```

检查 CUDA：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

必须看到 `True` 和实际 GPU 名称，否则不要启动训练。

## 6. 数据与文件检查

APT 默认读取根目录 `data/`。检查关键目录：

```bash
python -c "from pathlib import Path; print(Path('data').resolve()); print(Path('apt_experiments/experiments/scans').resolve())"
```

需要同步到 GPU 机器的内容：

- 整个 `apt_experiments/`
- 根目录 `datasets.py`
- 根目录 `data/`，或允许脚本在 GPU 机器重新下载
- 需要做最终对照的 Full ViT、MAE + Router checkpoint

不需要把旧 APT checkpoint 混入 `apt_experiments/checkpoints/`。

## 7. 生成第一轮队列

```bash
python apt_experiments/scripts/generate_gpu_experiments.py
```

检查任务数：

```bash
python -c "import json; print(json.load(open('apt_experiments/experiments/queues/manifest.json')))"
```

正常情况下 `queue_size` 为 `6`，`datasets_missing_scans` 为空。

## 8. 启动第一轮筛选

Linux：

```bash
bash apt_experiments/experiments/queues/gpu_short_screen.sh
```

Windows PowerShell：

```powershell
.\apt_experiments\experiments\queues\gpu_short_screen.ps1
```

相同命令再次运行时，训练脚本会从
`apt_experiments/checkpoints/<run_name>/checkpoint_epoch_*.pth` 自动续训。

RTX 4090 上第一轮预计约 `5-9` 小时。当前训练使用 FP32，Hierarchical APT 含逐图
Python 处理，因此实际时间受 CPU 与磁盘性能影响较大。

## 9. 汇总第一轮结果

```bash
python apt_experiments/scripts/aggregate_experiment_results.py
```

输出：

```text
apt_experiments/experiments/results/results.csv
apt_experiments/experiments/results/results.json
```

第一轮淘汰规则：

1. 预扫描 token 减少不足 `20%`：淘汰。
2. 5 epochs validation accuracy 明显低于其他 APT 方法：淘汰。
3. 运行异常、token 分布退化到最小值或 padding 异常：修复后仅重跑受影响任务。
4. 每个数据集最多保留 1 至 2 个方法。

## 10. 第二轮与正式训练

生成包含 75% 和 50% 预算的候选队列：

```bash
python apt_experiments/scripts/generate_gpu_experiments.py --ratios 0.75 0.5 --short_epochs 10
```

该命令会重新生成全部三种主方案。实际执行前应从 JSONL 中只保留第一轮胜出方案，
不要无差别运行整个矩阵。

正式 epoch：

- CIFAR-100：最多 100 epochs。
- Oxford Pets：最多 100 epochs。
- Food-101：如扩展，最多 30 epochs。
- DTD：如扩展，最多 100 epochs。

正式训练至少使用 seeds `42` 和 `3407`。候选配置冻结后才能执行最终 test。

## 11. 最终资源验收

最终必须在同一张 GPU 上比较：

- Full ViT-B/16；
- APT 正式候选；
- 可选参考：MAE + Router。

统一记录：

- test accuracy；
- 平均真实 token 与 padded token；
- batch size 1 端到端延迟；
- batch size 16 或 32 吞吐量；
- 峰值 GPU 显存；
- entropy/tokenization 在内的总耗时。

当前第一轮队列会产生 APT 的准确率、延迟和吞吐量，但它不等于最终验收，原因是：

1. 只有 5 epochs 和单 seed；
2. 尚未在同一 GPU 上重测 Full ViT；
3. 当前结果文件尚未统一写入峰值显存；
4. 第一轮只用于筛选，不能用 test 结果反向选择候选。

因此，运行第一轮脚本后还不能直接得出最终结论。最终结论必须在候选冻结、完整训练并完成
统一资源 benchmark 后给出。

## 12. OOM 与中断处理

若 24 GB 显存仍 OOM：

1. 将 `--batch_size 16 --accum 8` 改为 `--batch_size 8 --accum 16`；
2. 保持 effective batch size 为 128；
3. 不改变阈值、seed、学习率和数据划分；
4. 重新运行相同命令，自动续训。

租赁实例中断前同步整个目录：

```text
apt_experiments/checkpoints/
apt_experiments/experiments/results/
```
