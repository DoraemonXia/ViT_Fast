# APT 精简路线

## 主线

1. GPU 完整训练 A4 Learned Hierarchical APT。
2. 数据集：CIFAR-100、Oxford Pets。
3. 默认约 75% token，seed 42。
4. 自动保存 checkpoint、history 和 results。
5. 汇总结果并与历史 Fixed Merge 对比。

## 备用

A3 Average Hierarchical APT 位于：

```text
apt_experiments/Hierarchical_16_32_Average_APT/train.py
```

默认不运行。只有 A4 出现训练异常或 learned aggregation 没有合理收益时启用。

## 验收

- A4 精度相对 Full ViT 的下降；
- A4 相对历史 Merge 的精度与 token 改善；
- 端到端延迟、吞吐量和峰值显存；
- 是否在减少 token 的同时保持可接受精度。


