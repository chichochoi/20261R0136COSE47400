"""Vanilla CVAE model definition.

This is the baseline model without U-Net skip connections and without
classifier-guided loss. The architecture is intentionally kept simple
to isolate the effect of those two components in the main experiment.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def label_to_onehot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).float()


def make_label_map(
    labels: torch.Tensor,
    num_classes: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Create spatial label maps for encoder conditioning.

    Image shape:     [B, 1, H, W]
    Label map shape: [B, C, H, W]
    Combined input:  [B, 1 + C, H, W]
    """
    onehot = label_to_onehot(labels, num_classes)
    return onehot[:, :, None, None].repeat(1, 1, height, width)


class VanillaCVAE(nn.Module):
    """Baseline Conditional VAE without skip connections or classifier guidance.

    The encoder receives a grayscale MRI slice concatenated with a spatial
    label map. The decoder receives the latent vector concatenated with a
    one-hot class vector. No skip connections are used.
    """

    def __init__(
        self,
        img_size: int = 224,
        channels: int = 1,
        num_classes: int = 3,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.channels = channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        encoder_input_channels = channels + num_classes
        self.encoder = nn.Sequential(
            nn.Conv2d(encoder_input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.feature_size = img_size // 16
        self.flatten_dim = 256 * self.feature_size * self.feature_size
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim + num_classes, self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor, labels: torch.Tensor):
        batch_size, _, height, width = x.shape
        label_map = make_label_map(labels, self.num_classes, height, width).to(x.device)
        features = self.encoder(torch.cat([x, label_map], dim=1))
        features = features.view(batch_size, -1)
        return self.fc_mu(features), self.fc_logvar(features)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        onehot = label_to_onehot(labels, self.num_classes).to(z.device)
        x = self.decoder_input(torch.cat([z, onehot], dim=1))
        x = x.view(-1, 256, self.feature_size, self.feature_size)
        return self.decoder(x)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, labels)
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}
