$ErrorActionPreference = 'Stop'
python apt_experiments/train_apt_patch_selection.py --dataset cifar100 --gpu 0 --batch_size 16 --accum 8 --epochs 5 --seed 42 --entropy_bins 64 --threshold 2.0
python apt_experiments/train_apt_patch_merge.py --dataset cifar100 --gpu 0 --batch_size 16 --accum 8 --epochs 5 --seed 42 --entropy_bins 64 --threshold 3.25
python apt_experiments/train_hierarchical_apt.py --dataset cifar100 --gpu 0 --batch_size 16 --accum 8 --epochs 5 --seed 42 --entropy_bins 64 --threshold32 3.25 --aggregation average
python apt_experiments/train_apt_patch_selection.py --dataset oxford_pets --gpu 0 --batch_size 16 --accum 8 --epochs 5 --seed 42 --entropy_bins 64 --threshold 2.75
python apt_experiments/train_apt_patch_merge.py --dataset oxford_pets --gpu 0 --batch_size 16 --accum 8 --epochs 5 --seed 42 --entropy_bins 64 --threshold 4.0
python apt_experiments/train_hierarchical_apt.py --dataset oxford_pets --gpu 0 --batch_size 16 --accum 8 --epochs 5 --seed 42 --entropy_bins 64 --threshold32 4.0 --aggregation average
