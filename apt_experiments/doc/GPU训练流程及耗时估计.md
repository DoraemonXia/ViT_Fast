# GPU 训练流程及耗时估计

当前 GPU 主线只运行 A4 Learned Hierarchical APT。目标是避免“服务器到期、训练没跑完、结果没导出”导致白跑。

核心策略：

- 用 `tmux` 跑训练，SSH 或 VS Code Remote SSH 断开后训练仍继续。
- 训练脚本每个 epoch 自动保存 checkpoint。
- 定期把 checkpoint 和日志下载回本机。
- 只要本机保存了 `checkpoint_epoch_*.pth` 或 `best_model.pth`，换服务器后就能继续跑。

## 1. VS Code Remote SSH 连接服务器

在 VS Code 中：

1. 点击左下角绿色远程连接按钮。
2. 选择 `Connect to Host...`。
3. 连接你的服务器。
4. 打开服务器上的项目目录，例如：

```bash
cd /workspace/ViT_Fast
```

如果是 AutoDL，项目路径可能是：

```bash
cd /root/autodl-tmp/ViT_Fast
```

后续除特别说明外，所有 Linux 命令都在服务器项目根目录执行。

## 2. 检查 GPU 和环境

先确认 GPU 可见：

```bash
nvidia-smi
```

确认 Python 可以使用 CUDA：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

必须看到 `True` 和 GPU 名称。

安装依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install -r apt_experiments/requirements.txt
```

如果服务器环境已经配好，可以先快速检查：

```bash
python -c "import torch, torchvision, timm, tqdm; print('env ok')"
```

### 2.1 准备离线 ViT-B/16 预训练权重

A4 默认使用 `vit_base_patch16_224.augreg_in21k`。`timm` 会尝试从 Hugging Face
下载权重；AutoDL 无法访问 Hugging Face 时会在训练开始前报
`Cannot assign requested address`。

正式实验不要用 `--no_pretrained` 绕过。该参数会让模型随机初始化，不能与原来的
IN-21K 预训练基线公平比较。

推荐在能访问 Hugging Face 的本机或其他机器上生成权重文件：

```bash
python - <<'PY'
import timm
import torch

model = timm.create_model(
    "vit_base_patch16_224.augreg_in21k",
    pretrained=True,
    num_classes=0,
)
torch.save(
    {"state_dict": model.state_dict()},
    "vit_base_patch16_224_augreg_in21k.pth",
)
print("saved: vit_base_patch16_224_augreg_in21k.pth")
PY
```

然后在本机 PowerShell 上传到服务器：

```powershell
scp -P 12345 "C:\path\to\vit_base_patch16_224_augreg_in21k.pth" `
  root@HOST:/root/autodl-tmp/ViT_Fast/apt_experiments/pretrained/
```

在服务器项目根目录检查：

```bash
cd /root/autodl-tmp/ViT_Fast
test -f apt_experiments/pretrained/vit_base_patch16_224_augreg_in21k.pth
echo $?
```

输出 `0` 表示文件存在。训练脚本传入 `--pretrained_checkpoint` 后不会再访问
Hugging Face。

## 3. 准备 CIFAR-100 数据集

训练第一次运行时会下载 CIFAR-100。Toronto 源有时非常慢，如果服务器下载速度只有几 KB/s，直接停止下载，手动上传数据集。

服务器训练窗口按：

```text
Ctrl + C
```

在本机浏览器下载：

```text
https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
```

然后在本机 PowerShell 上传到服务器项目的 `data/` 目录。

普通 SSH：

```powershell
scp "C:\Users\32725\Downloads\cifar-100-python.tar.gz" `
  USER@SERVER:/root/autodl-tmp/ViT_Fast/data/
```

如果 SSH 有端口，例如 `ssh root@HOST -p 12345`，则用：

```powershell
scp -P 12345 "C:\Users\32725\Downloads\cifar-100-python.tar.gz" `
  root@HOST:/root/autodl-tmp/ViT_Fast/data/
```

服务器上确认文件存在且约 169MB：

```bash
cd /root/autodl-tmp/ViT_Fast
ls -lh data/cifar-100-python.tar.gz
```

可选：先验证 torchvision 能识别数据集：

```bash
python - <<'PY'
from torchvision import datasets
datasets.CIFAR100(root='./data', train=True, download=True)
print('cifar100 ok')
PY
```

如果输出 `Files already downloaded and verified`，训练时不会重新下载。

## 4. 开启 tmux

如果服务器没有 `tmux`，先安装：

```bash
sudo apt update
sudo apt install tmux -y
```

新建训练会话：

```bash
tmux new -s apt
```

进入后，你就在 `tmux` 会话里了。后面的训练命令建议都在这个会话里执行。

## 5. 启动 CIFAR-100 A4 训练

推荐先跑 CIFAR-100 A4：

```bash
cd /root/autodl-tmp/ViT_Fast
test -f apt_experiments/Hierarchical_16_32_Learned_APT/train.py
test -f apt_experiments/pretrained/vit_base_patch16_224_augreg_in21k.pth

python -u apt_experiments/Hierarchical_16_32_Learned_APT/train.py \
  --dataset cifar100 \
  --gpu 0 \
  --batch_size 16 \
  --accum 8 \
  --epochs 100 \
  --seed 42 \
  --entropy_bins 64 \
  --threshold32 3.25 \
  --pretrained_checkpoint apt_experiments/pretrained/vit_base_patch16_224_augreg_in21k.pth \
  2>&1 | tee -a apt_experiments/a4_cifar100.log
```

说明：

- 命令必须从 `/root/autodl-tmp/ViT_Fast` 项目根目录运行。
- `--pretrained_checkpoint` 使用服务器本地权重，不再联网下载。
- `python -u`：实时刷新日志。
- `tee -a`：屏幕能看，日志也写入 `apt_experiments/a4_cifar100.log`。
- 训练和验证都会显示 `tqdm` 进度条。
- 每个 epoch 结束后保存 checkpoint。
- 第一个 epoch 结束前没有完整恢复点；至少等 `checkpoint_epoch_0.pth` 出现后才算有可恢复进度。

训练输出目录：

```text
apt_experiments/Hierarchical_16_32_Learned_APT/checkpoints/
└── cifar100_a4_learned_224_t32_3.25_s42/
    ├── args.json
    ├── history.json
    ├── checkpoint_epoch_*.pth
    ├── best_model.pth
    └── results.json
```

## 6. tmux 常用操作

让训练留在服务器后台继续跑：

```text
Ctrl + b
```

松开后再按：

```text
d
```

这叫 detach。VS Code 或 SSH 断开后，训练仍在服务器的 `tmux` 会话里跑。

重新进入训练会话：

```bash
tmux attach -t apt
```

查看已有会话：

```bash
tmux ls
```

查看日志：

```bash
tail -f apt_experiments/a4_cifar100.log
```

## 7. 定期备份到本机

这是最重要的一步。不要等服务器快到期才下载。

下面命令在**本机 PowerShell** 中执行，不是在服务器里执行。

普通 SSH：

```powershell
scp -r USER@SERVER:/root/autodl-tmp/ViT_Fast/apt_experiments/Hierarchical_16_32_Learned_APT/checkpoints `
  "C:\Users\32725\Desktop\apt_backup\"

scp -r USER@SERVER:/root/autodl-tmp/ViT_Fast/apt_experiments/a4_cifar100.log `
  "C:\Users\32725\Desktop\apt_backup\"
```

带端口 SSH：

```powershell
scp -P 12345 -r root@HOST:/root/autodl-tmp/ViT_Fast/apt_experiments/Hierarchical_16_32_Learned_APT/checkpoints `
  "C:\Users\32725\Desktop\apt_backup\"

scp -P 12345 -r root@HOST:/root/autodl-tmp/ViT_Fast/apt_experiments/a4_cifar100.log `
  "C:\Users\32725\Desktop\apt_backup\"
```

需要替换：

- `USER@SERVER` 或 `root@HOST`：你的服务器用户名和地址。
- `12345`：你的 SSH 端口。
- `/root/autodl-tmp/ViT_Fast`：服务器上的真实项目路径。
- `C:\Users\32725\Desktop\apt_backup\`：你本机想保存备份的位置。

建议备份频率：

- 第 1 个 epoch 结束后立刻备份一次。
- 之后每 30-60 分钟备份一次。
- 服务器快到期前必须再备份一次。

## 8. 服务器快到期但训练没跑完

先不要慌，按下面顺序做。

第一，服务器上确认 checkpoint 已经存在：

```bash
ls -lh apt_experiments/Hierarchical_16_32_Learned_APT/checkpoints/cifar100_a4_learned_224_t32_3.25_s42/
```

第二，立刻在本机 PowerShell 下载 checkpoint：

```powershell
scp -r USER@SERVER:/root/autodl-tmp/ViT_Fast/apt_experiments/Hierarchical_16_32_Learned_APT/checkpoints `
  "C:\Users\32725\Desktop\apt_backup\"
```

第三，如果已经生成汇总结果，也下载：

```powershell
scp -r USER@SERVER:/root/autodl-tmp/ViT_Fast/apt_experiments/experiments/results `
  "C:\Users\32725\Desktop\apt_backup\"
```

如果训练没跑完整，`results.json` 可能还没有。此时最关键的是：

```text
checkpoint_epoch_*.pth
best_model.pth
history.json
a4_cifar100.log
```

只要这些已经下载回本机，就不算白跑。

## 9. 换服务器继续跑

把本机备份上传回新服务器同样路径。

在本机 PowerShell 中执行：

```powershell
scp -r "C:\Users\32725\Desktop\apt_backup\checkpoints" `
  USER@NEW_SERVER:/root/autodl-tmp/ViT_Fast/apt_experiments/Hierarchical_16_32_Learned_APT/
```

然后在新服务器项目根目录重新执行同一条训练命令：

```bash
python -u apt_experiments/Hierarchical_16_32_Learned_APT/train.py \
  --dataset cifar100 \
  --gpu 0 \
  --batch_size 16 \
  --accum 8 \
  --epochs 100 \
  --seed 42 \
  --entropy_bins 64 \
  --threshold32 3.25 \
  --pretrained_checkpoint apt_experiments/pretrained/vit_base_patch16_224_augreg_in21k.pth \
  2>&1 | tee -a apt_experiments/a4_cifar100.log
```

脚本会自动查找最新的：

```text
checkpoint_epoch_*.pth
```

如果恢复成功，日志里会看到：

```text
[RESUME] Loading checkpoint: ...
```

注意：自动恢复只能从已完成的 epoch 继续。如果中断发生在 epoch 中间，会从上一个完整 checkpoint 恢复。

## 10. 训练完成后生成最终结果

训练完整跑完后，在服务器项目根目录执行：

```bash
python apt_experiments/scripts/aggregate_experiment_results.py
```

生成：

```text
apt_experiments/experiments/results/results.csv
apt_experiments/experiments/results/results.json
apt_experiments/experiments/results/RESULTS_TABLE.md
```

下载最终结果到本机：

```powershell
scp -r USER@SERVER:/root/autodl-tmp/ViT_Fast/apt_experiments/experiments/results `
  "C:\Users\32725\Desktop\apt_backup\"
```

最终建议保留：

```text
checkpoints/
a4_cifar100.log
experiments/results/results.csv
experiments/results/results.json
experiments/results/RESULTS_TABLE.md
```

## 11. A3 备用命令

A3 不进入默认队列。只有 A4 训练异常或 learned aggregation 没有收益时运行：

```bash
python -u apt_experiments/Hierarchical_16_32_Average_APT/train.py \
  --dataset cifar100 \
  --gpu 0 \
  --batch_size 16 \
  --accum 8 \
  --epochs 100 \
  --seed 42 \
  --entropy_bins 64 \
  --threshold32 3.25 \
  2>&1 | tee -a apt_experiments/a3_cifar100.log
```

## 12. 耗时估计

A4 包含逐图层次划分和 learned aggregation，当前使用 FP32。实际时间以服务器第一个 epoch 日志为准。

| 设备 | CIFAR-100 100 epochs | CIFAR-100 1 epoch | 备注 |
|:-----|:---------------------|:------------------|:-----|
| CPU | 不建议，可能数天级 | 可能数小时级 | 当前正式训练默认要求 CUDA |
| RTX 4090 | 12-22 小时 | 7-15 分钟 | 旧估计，按第一轮日志校准 |
| RTX 5090 | 8-16 小时 | 5-12 分钟 | 粗估，受环境和 IO 影响 |

只想检查脚本能否启动时，可以先生成短队列：

```bash
python apt_experiments/scripts/generate_gpu_experiments.py --epochs 5
```

5 epoch 只能验证链路，不能作为最终实验结论。

## 13. 最保险习惯

- 租服务器前确认：实例到期后系统盘和数据盘是否保留。
- 第 1 个 epoch 结束后先备份一次。
- 不要相信“等跑完再下载”。
- 如果平台会释放磁盘，必须定期下载 checkpoint。
- 只要本机保存了 `checkpoint_epoch_*.pth` 或 `best_model.pth`，这次训练就没有白跑。
