# Hierarchical 16/32 Average APT

A3 备用对照实验。`train.py` 是单文件训练入口，整体流程与 Learned APT 相同，但低熵 `32x32` 区域的粗 token 使用平均池化生成。

## 方法

- 高熵 `32x32` 区域拆成 4 个 `16x16` 细 token。
- 低熵 `32x32` 区域合并成 1 个粗 token。
- 粗 token 由 4 个子 token 直接平均得到。
- 保留 scale encoding 和 masked attention，方便与 A4 公平对比。

## 运行

```bash
python apt_experiments/Hierarchical_16_32_Average_APT/train.py --dataset cifar100 --gpu 0 --threshold32 3.25
```

训练输出默认写入本目录下的 `checkpoints/`。
