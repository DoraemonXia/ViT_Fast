# Experiment Results

This directory stores result records for the capstone token-reduction experiments.

## Fixed Reference Numbers

These values are copied from the existing project reports and are kept as context, not as new ToMe/EViT measurements.

| Group | Method | Tokens | Accuracy | Source |
| --- | --- | ---: | ---: | --- |
| Trained baseline | Full ViT-B/16 | 196/196 | 91.69% | `README.md`, `EXPERIMENTS.md` |
| Trained | MAE+Router keep75 | 147/196 | 91.10% | `README.md`, `EXPERIMENTS.md` |
| Trained | MAE+Router keep50 | 98/196 | 89.07% | `EXPERIMENTS.md` |
| Trained / finetuned | APT Entropy Selection | 133/196 | 85.84% | `EXPERIMENTS.md` |
| Trained / finetuned | APT Merge | 73/196 | 83.13% | `EXPERIMENTS.md` |

## Generated Files

`eval_training_free_token_reduction.py` appends new training-free results to:

- `training_free_token_reduction.csv`
- `training_free_token_reduction.jsonl`

Use the CSV for report tables. Use the JSONL when you need to recover the exact run parameters.

Keep training-free ToMe/EViT rows separate from trained APT/MAE+Router rows unless the ToMe/EViT setting is also finetuned.
