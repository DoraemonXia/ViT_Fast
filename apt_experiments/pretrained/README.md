# 本地预训练权重

将上传到 GPU 服务器的 ViT-B/16 IN-21K 权重放在本目录，例如：

```text
apt_experiments/pretrained/vit_base_patch16_224_augreg_in21k.pth
```

权重文件已被 Git 忽略。训练时传入：

```bash
--pretrained_checkpoint apt_experiments/pretrained/vit_base_patch16_224_augreg_in21k.pth
```
