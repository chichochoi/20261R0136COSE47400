# Vanilla CVAE Baseline

This folder contains the baseline vanilla CVAE experiment, aligned with the
main U-Net CVAE experiment for fair comparison.

## What Changed (Aligned Baseline)

The original notebook-based baseline had several experimental-setup differences
that made fair comparison impossible. This version fixes them:

| Setting | Old Baseline | Aligned Baseline | UNet CVAE |
|---------|-------------|-----------------|-----------|
| Data split | Random image-level | **Subject-based** | Subject-based |
| Epochs | 10 | **50** | 50 |
| Optimizer | Adam | **AdamW** (wd=1e-5) | AdamW (wd=1e-5) |
| Scheduler | None | **CosineAnnealing** | CosineAnnealing |
| Grad clipping | None | **max_norm=1.0** | max_norm=1.0 |
| KL warmup | None | **5 epochs** (linear) | N/A |
| Val set | 17,189 images | Same as UNet | Same as UNet |

The **model architecture is unchanged**: no skip connections, no classifier
guidance. These are the two features being tested by comparison.

## Contents

```text
vanilla/
  configs/default.yaml
  scripts/
    train_cvae.py
    evaluate.py
    smoke_test.py
  src/vanilla_cvae/
    data.py
    losses.py
    metrics.py
    models.py
    utils.py
  Baseline_Vanilla_CVAE.ipynb      # Legacy notebook (kept for reference)
  Vanilla_CVAE_Baseline.ipynb      # Legacy notebook (kept for reference)
```

## Setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 1. Smoke Test

Run this first. No data required.

```bash
python scripts/smoke_test.py
```

## 2. Train the Vanilla CVAE

```powershell
python scripts\train_cvae.py --data-dir data
```

The data directory should contain class folders directly:

```text
data/
  Non Demented/
  Very mild Dementia/
  Mild Dementia/
```

The saved model is:

```text
checkpoints/vanilla_cvae/best_vanilla_cvae.pth
```

## 3. Evaluate

Use the **same classifier checkpoint** as the UNet experiment for fair
comparison:

```powershell
python scripts\evaluate.py `
  --data-dir data `
  --cvae-checkpoint checkpoints\vanilla_cvae\best_vanilla_cvae.pth `
  --classifier-checkpoint ..\unet\checkpoints\classifier\best_classifier.pth
```

Reported metrics (identical to UNet evaluation):

- Reconstruction MSE
- PSNR
- SSIM
- Recon Loss (BCE), KL Loss, Val Loss
- Classifier target confidence on reconstructions
- Classifier accuracy on reconstructions
- Confusion matrix

## Model Definition

The CVAE encodes a grayscale MRI slice concatenated with a spatial label map.
The class label is provided as a one-hot vector concatenated with `z` at the
bottleneck before decoding. **No skip connections** are used.

Training loss:

```text
L = L_reconstruction + beta(t) * L_KL
```

Where `beta(t)` linearly warms up from 0 to 1.0 over the first 5 epochs.

This baseline intentionally omits:
- U-Net skip connections
- Classifier-guided loss

These are the two architectural features being evaluated by the main experiment.
