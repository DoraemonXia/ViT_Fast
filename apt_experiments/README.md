# APT Experiments

本目录是项目中 Adaptive Patch Transformer（APT）实验的复现入口。研究目标是验证：Vision Transformer 能否根据图像内容动态减少 Patch Token，同时尽量保持分类精度并降低实际推理开销。

当前主方案为 **Hierarchical 16/32 Learned APT（A4）**。Fixed Merge 和 A3 Average APT 用作对照，不属于默认主实验。

## 1. 方案总览

| 方法 | 入口 | Token 处理 | 用途 |
|:---|:---|:---|:---|
| A4 Learned APT | `Hierarchical_16_32_Learned_APT/train.py` | 低熵区域将 4 个 `16×16` Token 学习聚合为 1 个 `32×32` Token | 主实验 |
| A3 Average APT | `Hierarchical_16_32_Average_APT/train.py` | 与 A4 相同，但使用平均聚合 | 备用消融 |
| Fixed 16/32 Merge | `../train_apt_patch_merge.py` | 低熵区域固定平均合并 | 历史与速度对照 |
| Full ViT-B/16 | 根目录原始训练脚本 | 保留全部 196 Tokens | 精度和速度基线 |

A4 使用图像熵决定每个 `32×32` 区域采用粗粒度还是细粒度表示，并使用 Scale Encoding 区分不同尺度 Token。

## 2. 目录结构

```text
ViT_Fast/
├── apt_experiments/
│   ├── Hierarchical_16_32_Learned_APT/
│   │   ├── train.py                 # A4 主实验：训练、验证和测试
│   │   ├── README.md                # A4 方法说明
│   │   └── checkpoints/             # 运行时生成，Git 忽略
│   ├── Hierarchical_16_32_Average_APT/
│   │   ├── train.py                 # A3 平均聚合对照
│   │   ├── README.md
│   │   └── checkpoints/             # 运行时生成，Git 忽略
│   ├── pretrained/
│   │   ├── README.md
│   │   └── model.safetensors        # 本地 ViT-B/16 权重，Git 忽略
│   ├── scripts/
│   │   ├── scan_apt_thresholds.py   # CPU 阈值与 Token 数扫描
│   │   ├── generate_gpu_experiments.py
│   │   ├── benchmark_a4_speed.py    # Baseline/A4 同环境测速
│   │   └── aggregate_experiment_results.py
│   ├── experiments/
│   │   ├── scans/                   # 阈值扫描结果
│   │   ├── queues/                  # 自动生成的 GPU 命令
│   │   ├── references/              # 历史参考结果
│   │   └── results/                 # 自动汇总结果，Git 忽略
│   ├── results_downloaded/
│   │   ├── CIFAR100/                # 从服务器取回的结果
│   │   └── OxfordPets/              # 权重可留在本地，但被 Git 忽略
│   ├── benchmark_results/            # 同环境测速生成的 JSON
│   ├── tests/
│   │   └── test_apt_stage0.py       # CPU 单元与模型冒烟测试
│   ├── doc/
│   │   ├── GPU_WORKFLOW.md
│   │   ├── GPU训练流程及耗时估计.md
│   │   └── ROADMAP.md
│   ├── requirements.txt
│   └── README.md
├── data/                             # 数据集，Git 忽略
├── checkpoints/                      # Baseline/Merge 权重，Git 忽略
├── datasets.py                       # APT 复用的数据加载器
├── train_apt_patch_merge.py          # Fixed Merge 对照
└── APT实验结果分析.md
```

`train.py`、`datasets.py` 和模型权重路径存在相对引用，以下命令均应从仓库根目录 `ViT_Fast/` 执行。

## 3. 复现环境

推荐环境：

- Linux GPU 服务器；
- NVIDIA RTX 4090D 或同等级 GPU；
- Python 3.10 及以上；
- CUDA 可用的 PyTorch；
- 至少 50GB 可用磁盘，完整保留 A4 全部 Epoch 权重时建议 100GB 以上。

安装依赖：

```bash
cd /path/to/ViT_Fast
python -m pip install -r apt_experiments/requirements.txt
```

验证 CUDA：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

预期输出包含：

```text
True
NVIDIA GeForce RTX 4090 D
```

## 4. 准备数据集

正式结果使用以下目录：

```text
data/
├── cifar-100-python/
│   ├── train
│   ├── test
│   └── meta
└── oxford-iiit-pet/
    ├── images/
    └── annotations/
```

验证 CIFAR-100：

```bash
python -c "from torchvision.datasets import CIFAR100; d=CIFAR100(root='./data', train=True, download=False); print(len(d))"
```

预期输出为 `50000`。

验证 Oxford Pets：

```bash
python -c "from torchvision.datasets import OxfordIIITPet; d=OxfordIIITPet(root='./data', split='trainval', download=False); print(len(d))"
```

预期输出为 `3680`。如果服务器网络不稳定，应在本地下载并上传完整数据集，避免训练脚本在运行时联网。

## 5. 准备预训练权重

正式实验使用：

```text
apt_experiments/pretrained/model.safetensors
```

该文件对应 `timm/vit_base_patch16_224.augreg_in21k`，约 392MB。训练时必须传入 `--pretrained_checkpoint`，否则 `timm` 会尝试访问 Hugging Face。

检查文件：

```bash
ls -lh apt_experiments/pretrained/model.safetensors
python -c "from safetensors.torch import load_file; w=load_file('apt_experiments/pretrained/model.safetensors'); print(len(w)); print(next(iter(w)))"
```

当前文件应包含约 152 个 Tensor。分类头形状与目标数据集不同属于正常情况。

## 6. 训练前检查

加载 A4 模型和本地权重：

```bash
python -c "from apt_experiments.Hierarchical_16_32_Learned_APT.train import HierarchicalAPTViT; HierarchicalAPTViT(num_classes=100, pretrained_checkpoint='apt_experiments/pretrained/model.safetensors'); print('model load ok')"
```

可选 CPU 测试：

```bash
python -m unittest apt_experiments.tests.test_apt_stage0
```

该测试只用于检查熵计算、Attention Mask、Token 范围和数据划分，不替代 GPU 正式训练。

## 7. 复现 A4 正式实验

### 7.1 CIFAR-100

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

输出目录：

```text
apt_experiments/Hierarchical_16_32_Learned_APT/checkpoints/
└── cifar100_a4_learned_224_t32_3.25_s42/
    ├── args.json
    ├── history.json
    ├── checkpoint_epoch_*.pth
    ├── best_model.pth
    └── results.json
```

### 7.2 Oxford Pets

```bash
python -u apt_experiments/Hierarchical_16_32_Learned_APT/train.py \
  --dataset oxford_pets \
  --gpu 0 \
  --batch_size 32 \
  --accum 4 \
  --epochs 70 \
  --seed 42 \
  --entropy_bins 64 \
  --threshold32 4.0 \
  --pretrained_checkpoint apt_experiments/pretrained/model.safetensors \
  --num_workers 8 \
  --log_interval 20 \
  2>&1 | tee -a apt_experiments/a4_oxford_pets.log
```

输出目录：

```text
apt_experiments/Hierarchical_16_32_Learned_APT/checkpoints/
└── oxford_pets_a4_learned_224_t32_4.0_s42/
```

### 7.3 后台运行与恢复

建议在 `tmux` 中运行：

```bash
tmux new -s apt
```

使用 `Ctrl+B`、再按 `D` 可退出并保持训练。重新连接后执行：

```bash
tmux attach -t apt
```

训练脚本会自动查找输出目录中编号最大的 `checkpoint_epoch_*.pth` 并恢复，也可显式指定：

```bash
--resume /path/to/checkpoint_epoch_XX.pth
```

每个完整 A4 Checkpoint 约 985MB。长时间训练前应执行 `df -h` 检查磁盘；清理旧权重时至少保留：

- `best_model.pth`；
- 最新且确认可读取的 `checkpoint_epoch_*.pth`；
- `args.json`、`history.json` 和 `results.json`。

## 8. 复现 Fixed Merge 对照

Fixed Merge 位于仓库根目录：

```bash
python -u train_apt_patch_merge.py \
  --dataset cifar100 \
  --gpu 0 \
  --batch_size 32 \
  --accum 4 \
  --epochs 50 \
  --seed 42 \
  --threshold 5.5 \
  --pretrained_checkpoint apt_experiments/pretrained/model.safetensors \
  --num_workers 8
```

输出目录：

```text
checkpoints/cifar100_apt_merge_t5.5_s42/
├── args.json
├── history.json
├── checkpoint_latest.pth
├── best_model.pth
└── results.json
```

该脚本滚动覆盖 `checkpoint_latest.pth`，不会按 Epoch 无限累积权重。

## 9. 同环境速度测评

速度比较必须让 Baseline 和 A4 使用相同 GPU、Batch Size、BF16 配置、预热次数和测量批次数。测速只执行推理，不会重新训练。

先准备：

```text
checkpoints/cifar100_vit_b16_ft
```

然后执行：

```bash
python -u apt_experiments/scripts/benchmark_a4_speed.py \
  --dataset cifar100 \
  --apt_checkpoint apt_experiments/Hierarchical_16_32_Learned_APT/checkpoints/cifar100_a4_learned_224_t32_3.25_s42/checkpoint_epoch_79.pth \
  --baseline_checkpoint checkpoints/cifar100_vit_b16_ft \
  --gpu 0 \
  --batch_size 32 \
  --warmup_batches 20 \
  --max_batches 100 \
  --repeats 3 \
  --measure_accuracy
```

结果保存到：

```text
apt_experiments/benchmark_results/cifar100_a4_speed.json
```

报告速度时应优先使用该脚本生成的同环境结果，不应直接比较不同 GPU 上的绝对吞吐量。

## 10. 辅助脚本

### 扫描阈值

```bash
python apt_experiments/scripts/scan_apt_thresholds.py \
  --dataset cifar100 \
  --max_samples 1000 \
  --thresholds 0.0:6.0:0.25
```

输出位于 `apt_experiments/experiments/scans/`。该步骤只估算不同阈值对应的 Token 数，不训练模型。

### 生成 GPU 命令

```bash
python apt_experiments/scripts/generate_gpu_experiments.py \
  --datasets cifar100 oxford_pets \
  --ratios 0.75 \
  --pretrained_checkpoint apt_experiments/pretrained/model.safetensors
```

输出位于 `apt_experiments/experiments/queues/`。

### 汇总结果

```bash
python apt_experiments/scripts/aggregate_experiment_results.py
```

输出：

```text
apt_experiments/experiments/results/
├── results.json
├── results.csv
└── RESULTS_TABLE.md
```

该脚本默认只扫描 `apt_experiments/` 内的 `results.json` 和 `experiments/references/`。根目录 `checkpoints/` 下的 Fixed Merge 新结果不会自动进入汇总，需要单独引用或先复制其轻量 `results.json`。

## 11. 结果文件含义

| 文件 | 内容 | 是否需要保留 |
|:---|:---|:---:|
| `args.json` | 完整命令参数 | 是 |
| `history.json` | 每轮训练、验证和 Token 统计 | 是 |
| `results.json` | 最佳模型测试精度、速度和显存摘要 | 是 |
| `best_model.pth` | 最佳验证精度模型 | 是 |
| `checkpoint_epoch_*.pth` | 模型、优化器和调度器状态 | 仅保留恢复所需版本 |
| `*.log` | 终端训练日志 | 建议保留或归档 |

模型权重和数据集已由 `.gitignore` 排除。轻量 JSON 结果可复制到 `results_downloaded/` 后提交，以便分析和复核。

## 12. 当前参考结果

| 方法 | 数据集 | Test Acc | Baseline | 平均真实 Tokens | 结论 |
|:---|:---|---:|---:|---:|:---|
| Fixed Merge | CIFAR-100 | 83.80% | 91.69% | 49.76/196 | 压缩高，但精度损失过大 |
| A4 Learned APT | CIFAR-100 | 91.07% | 91.69% | 145.24/196 | 精度损失 0.62% |
| A4 Learned APT | Oxford Pets | 93.00% | 93.81% | 140.34/196 | 精度损失 0.81% |

CIFAR-100 同环境 Benchmark 中，A4 相对 Full ViT 的吞吐量下降约 `10.8%`。原因是平均真实 Tokens 虽然减少约四分之一，但 Batch Padding 后仍约为 `191.7/196`，熵计算和动态 Token 构建开销超过了节省的计算。

完整分析见 [APT实验结果分析.md](../APT实验结果分析.md)。

## 13. 已知限制

1. A4 当前采用 Dense Padding，真实 Token 减少不等于同等比例的 FLOPs 或延迟减少。
2. A4 每轮保存完整训练 Checkpoint，长训练容易占满磁盘。
3. A3 目前没有接入 `--pretrained_checkpoint`，离线服务器运行时会依赖已有 `timm` 缓存或网络，因此不作为默认复现路径。
4. `--no_pretrained` 只用于代码冒烟测试，不应产生正式对比结果。
5. 不同脚本内置测速流程可能不同；正式加速比应使用 `benchmark_a4_speed.py` 统一测量。

更详细的服务器操作、上传下载和耗时说明见 [GPU训练流程及耗时估计.md](doc/GPU训练流程及耗时估计.md)。
