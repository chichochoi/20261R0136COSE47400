"""Train the Vanilla CVAE baseline.

Training setup is aligned with the main U-Net CVAE experiment for fair
comparison:
  - Same subject-based data split (no data leakage)
  - Same optimizer: AdamW with weight decay 1e-5
  - Same scheduler: CosineAnnealingLR
  - Same gradient clipping: max_norm 1.0
  - Same number of epochs: 50
  - KL warmup over first 5 epochs to prevent KL explosion

The only intentional differences from UNet CVAE are:
  - No U-Net skip connections (vanilla encoder-decoder)
  - No classifier-guided loss
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vanilla_cvae.data import CLASS_TO_INDEX, MRISliceDataset
from vanilla_cvae.losses import kl_warmup_beta, vanilla_cvae_loss
from vanilla_cvae.models import VanillaCVAE
from vanilla_cvae.utils import AverageMeter, ensure_dir, get_device, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Vanilla CVAE baseline.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="checkpoints/vanilla_cvae")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta-kl", type=float, default=1.0)
    parser.add_argument("--kl-warmup-epochs", type=int, default=5,
                        help="Linearly ramp KL weight from 0 to beta-kl over this many epochs.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def run_epoch(model, loader, optimizer, beta_kl: float, device, train: bool):
    model.train(train)
    loss_meter = AverageMeter()
    recon_meter = AverageMeter()
    kl_meter = AverageMeter()

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels, _, _ in tqdm(loader, desc="train" if train else "val"):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images, labels)
            losses = vanilla_cvae_loss(
                recon=output["recon"],
                target=images,
                mu=output["mu"],
                logvar=output["logvar"],
                beta_kl=beta_kl,
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            batch_size = images.size(0)
            loss_meter.update(losses["total"].item(), batch_size)
            recon_meter.update(losses["reconstruction"].item(), batch_size)
            kl_meter.update(losses["kl"].item(), batch_size)

    return {
        "loss": loss_meter.avg,
        "reconstruction": recon_meter.avg,
        "kl": kl_meter.avg,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    output_dir = ensure_dir(args.output_dir)

    train_dataset = MRISliceDataset(
        args.data_dir,
        split="train",
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        normalize=False,
    )
    val_dataset = MRISliceDataset(
        args.data_dir,
        split="val",
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        normalize=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = VanillaCVAE(
        img_size=args.image_size,
        channels=1,
        num_classes=len(CLASS_TO_INDEX),
        latent_dim=args.latent_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    best_val = float("inf")

    print(f"Device: {device}")
    print(f"Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")
    print(f"KL warmup: {args.kl_warmup_epochs} epochs | Target beta: {args.beta_kl}")

    for epoch in range(1, args.epochs + 1):
        current_beta = kl_warmup_beta(epoch, args.kl_warmup_epochs, args.beta_kl)

        train_stats = run_epoch(model, train_loader, optimizer, current_beta, device, train=True)
        val_stats = run_epoch(model, val_loader, optimizer, current_beta, device, train=False)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"beta {current_beta:.3f} | "
            f"train {train_stats['loss']:.4f} | "
            f"val {val_stats['loss']:.4f} "
            f"(rec {val_stats['reconstruction']:.4f}, kl {val_stats['kl']:.4f})"
        )

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(
                output_dir / "best_vanilla_cvae.pth",
                model_state=model.state_dict(),
                epoch=epoch,
                val_loss=best_val,
                image_size=args.image_size,
                latent_dim=args.latent_dim,
                num_classes=len(CLASS_TO_INDEX),
                beta_kl=args.beta_kl,
                classes=CLASS_TO_INDEX,
            )
            print(f"  Saved best CVAE: {output_dir / 'best_vanilla_cvae.pth'}")


if __name__ == "__main__":
    main()
