# APT Experiments

这是从原 ViT/MAE 工程中隔离出的 APT 实验子工程。

- GPU 全流程与候选方案：`GPU_WORKFLOW.md`
- 阶段路线图：`ROADMAP.md`
- CPU 扫描结果：`experiments/scans/`
- GPU 队列：`experiments/queues/`
- APT checkpoint：`checkpoints/`

所有命令从仓库根目录执行，例如：

```bash
python apt_experiments/scripts/generate_gpu_experiments.py
bash apt_experiments/experiments/queues/gpu_short_screen.sh
```
