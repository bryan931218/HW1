# CV2026 HW1

## Introduction

This project solves a 100-class image classification task for the Visual Recognition using Deep Learning course homework. The final implementation uses a ResNet-50 backbone with pretrained ImageNet weights, 5-fold cross-validation, exponential moving average (EMA), MixUp augmentation, and test-time augmentation (TTA) for robust inference.

## Environment Setup

Recommended environment:

- Python 3.10+
- PyTorch 2.2+
- torchvision
- numpy
- pandas
- matplotlib
- Pillow
- tqdm

Example setup with Conda:

```bash
conda create -n CVHW1 python=3.10
conda activate CVHW1
pip install torch torchvision numpy pandas matplotlib pillow tqdm
```

## Usage

Train the 5-fold ResNet-50 pipeline:

```bash
python train.py
```

Generate `prediction.csv` and `prediction.zip` from trained fold checkpoints:

```bash
python train.py --predict-only
```

## Performance Snapshot

