"""Smoke test for the aligned Vanilla CVAE baseline.

Verifies model construction, forward pass, loss computation,
and KL warmup without requiring any data files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vanilla_cvae.losses import kl_warmup_beta, vanilla_cvae_loss
from vanilla_cvae.models import VanillaCVAE


def main() -> None:
    img_size = 224
    channels = 1
    num_classes = 3
    latent_dim = 128

    model = VanillaCVAE(
        img_size=img_size,
        channels=channels,
        num_classes=num_classes,
        latent_dim=latent_dim,
    )

    x = torch.rand(2, channels, img_size, img_size)
    labels = torch.tensor([0, num_classes - 1], dtype=torch.long)
    output = model(x, labels)

    recon = output["recon"]
    mu = output["mu"]
    logvar = output["logvar"]

    assert recon.shape == x.shape, f"Recon shape mismatch: {recon.shape} vs {x.shape}"
    assert mu.shape == (2, latent_dim), f"Mu shape mismatch: {mu.shape}"
    assert logvar.shape == (2, latent_dim), f"Logvar shape mismatch: {logvar.shape}"

    # Test loss computation
    losses = vanilla_cvae_loss(recon, x, mu, logvar, beta_kl=1.0)
    losses["total"].backward()
    assert "total" in losses
    assert "reconstruction" in losses
    assert "kl" in losses

    # Test KL warmup
    assert kl_warmup_beta(0, warmup_epochs=5, target_beta=1.0) == 0.0
    assert kl_warmup_beta(3, warmup_epochs=5, target_beta=1.0) == 0.6
    assert kl_warmup_beta(5, warmup_epochs=5, target_beta=1.0) == 1.0
    assert kl_warmup_beta(10, warmup_epochs=5, target_beta=1.0) == 1.0

    print("Vanilla CVAE smoke test passed.")
    print(f"Image size: {img_size} | Classes: {num_classes} | Latent dim: {latent_dim}")
    print(f"Loss: {losses['total'].item():.4f} | Recon: {losses['reconstruction'].item():.4f} | KL: {losses['kl'].item():.4f}")
    print("KL warmup schedule verified.")
    print()
    print("Training setup (aligned with UNet CVAE):")
    print("  Optimizer:       AdamW (weight_decay=1e-5)")
    print("  Scheduler:       CosineAnnealingLR")
    print("  Grad clipping:   max_norm=1.0")
    print("  Epochs:          50")
    print("  KL warmup:       5 epochs (linear ramp)")
    print("  Data split:      Subject-based stratified (same as UNet)")


if __name__ == "__main__":
    main()
