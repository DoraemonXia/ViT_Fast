# Hierarchical 16/32 Learned APT

当前 APT 主实验。`train.py` 是单文件训练入口，包含熵值计算、层次化 token 构建、learned aggregator、masked attention、训练循环、评估和结果保存。

## 方法

- 高熵 `32x32` 区域拆成 4 个 `16x16` 细 token。
- 低熵 `32x32` 区域合并成 1 个粗 token。
- 粗 token 由轻量 learned aggregator 对 4 个子 token 加权聚合。
- 使用 scale encoding 区分不同尺度 token。

## 运行

正式实验使用本地 IN-21K 预训练权重，避免 GPU 服务器无法访问 Hugging Face：

```bash
python apt_experiments/Hierarchical_16_32_Learned_APT/train.py \
  --dataset cifar100 \
  --gpu 0 \
  --threshold32 3.25 \
  --pretrained_checkpoint apt_experiments/pretrained/vit_base_patch16_224_augreg_in21k.pth
```

`--no_pretrained` 仅用于启动检查，不用于正式对比实验。

训练输出默认写入本目录下的 `checkpoints/`。
