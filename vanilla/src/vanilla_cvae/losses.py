"""Loss functions for Vanilla CVAE baseline.

The baseline intentionally uses only BCE reconstruction loss and KL divergence,
without classifier-guided loss. A KL warmup schedule is provided to prevent
the KL explosion observed in the original training.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def normalize_for_classifier(
    images: torch.Tensor,
    mean: float = 0.456,
    std: float = 0.224,
) -> torch.Tensor:
    """Normalize images for the frozen classifier (same as UNet)."""
    return (images - mean) / std


def kl_warmup_beta(epoch: int, warmup_epochs: int = 5, target_beta: float = 1.0) -> float:
    """Linearly ramp beta from 0 to target_beta over warmup_epochs.

    This prevents KL explosion in early training by gradually introducing
    the KL regularization term.
    """
    if warmup_epochs <= 0:
        return target_beta
    return min(1.0, epoch / warmup_epochs) * target_beta


def vanilla_cvae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta_kl: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute vanilla CVAE loss: BCE reconstruction + beta * KL divergence.

    No classifier-guided loss is included — this is intentional to serve
    as a baseline for measuring the effect of classifier guidance.
    """
    reconstruction = F.binary_cross_entropy(recon, target, reduction="mean")
    kl = kl_divergence(mu, logvar)
    total = reconstruction + beta_kl * kl
    return {
        "total": total,
        "reconstruction": reconstruction.detach(),
        "kl": kl.detach(),
    }
