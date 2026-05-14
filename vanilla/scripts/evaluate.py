"""Evaluate a trained Vanilla CVAE.

Reports the same metrics as the main U-Net CVAE evaluation script:
  - MSE, PSNR, SSIM
  - Reconstruction Loss (BCE), KL Loss, Val Loss
  - Classifier target confidence, accuracy, and confusion matrix
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Also add UNet src path so we can optionally reuse the same classifier
UNET_ROOT = Path(__file__).resolve().parents[2] / "unet"
sys.path.insert(0, str(UNET_ROOT / "src"))

from vanilla_cvae.data import CLASS_TO_INDEX, MRISliceDataset
from vanilla_cvae.losses import kl_divergence, normalize_for_classifier
from vanilla_cvae.metrics import format_confusion_matrix, mse, psnr, ssim, update_confusion_matrix
from vanilla_cvae.models import VanillaCVAE
from vanilla_cvae.utils import AverageMeter, get_device, load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Vanilla CVAE.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--cvae-checkpoint", type=str, required=True)
    parser.add_argument("--classifier-checkpoint", type=str, default=None,
                        help="Path to frozen classifier checkpoint. Can be the same one used by UNet.")
    parser.add_argument("--split", type=str, choices=["train", "val", "all"], default="val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mean", type=float, default=0.456)
    parser.add_argument("--std", type=float, default=0.224)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def load_cvae(path: str, device: torch.device) -> VanillaCVAE:
    checkpoint = load_checkpoint(path, device)
    model = VanillaCVAE(
        img_size=int(checkpoint.get("image_size", 224)),
        channels=1,
        num_classes=int(checkpoint.get("num_classes", len(CLASS_TO_INDEX))),
        latent_dim=int(checkpoint.get("latent_dim", 128)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_classifier(path: str | None, device: torch.device):
    """Load the frozen SmallMRIClassifier.

    Tries to import from the UNet package first; if unavailable, defines
    the same architecture inline.
    """
    if path is None:
        return None
    checkpoint = load_checkpoint(path, device)
    num_classes = int(checkpoint.get("num_classes", len(CLASS_TO_INDEX)))

    try:
        from mid_unet_cvae.models import SmallMRIClassifier
    except ImportError:
        # Define inline if UNet package is not on the path
        import torch.nn as nn

        def conv_down(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        class SmallMRIClassifier(nn.Module):
            def __init__(self, num_classes=3):
                super().__init__()
                self.features = nn.Sequential(
                    conv_down(1, 32), conv_down(32, 64),
                    conv_down(64, 128), conv_down(128, 256),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.head = nn.Sequential(
                    nn.Flatten(), nn.Dropout(p=0.2),
                    nn.Linear(256, num_classes),
                )

            def forward(self, x):
                return self.head(self.features(x))

    classifier = SmallMRIClassifier(num_classes=num_classes).to(device)
    # Support both checkpoint key formats
    state_key = "model_state" if "model_state" in checkpoint else "model_state_dict"
    classifier.load_state_dict(checkpoint[state_key])
    classifier.eval()
    return classifier


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    cvae = load_cvae(args.cvae_checkpoint, device)
    classifier = load_classifier(args.classifier_checkpoint, device)
    dataset = MRISliceDataset(
        args.data_dir,
        split=args.split,
        image_size=cvae.img_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        normalize=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    mse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    bce_meter = AverageMeter()
    kl_meter = AverageMeter()
    confidence_meter = AverageMeter()
    correct = 0
    total = 0
    confusion = torch.zeros(len(CLASS_TO_INDEX), len(CLASS_TO_INDEX), dtype=torch.long)

    for images, labels, _, _ in tqdm(loader, desc="evaluate"):
        images = images.to(device)
        labels = labels.to(device)
        output = cvae(images, labels)
        recon = output["recon"]
        batch_size = images.size(0)

        mse_meter.update(mse(recon, images).item(), batch_size)
        psnr_meter.update(psnr(recon, images).item(), batch_size)
        ssim_meter.update(ssim(recon, images).item(), batch_size)

        recon_bce = F.binary_cross_entropy(recon, images, reduction="mean")
        kl = kl_divergence(output["mu"], output["logvar"])
        bce_meter.update(recon_bce.item(), batch_size)
        kl_meter.update(kl.item(), batch_size)

        if classifier is not None:
            logits = classifier(normalize_for_classifier(recon, args.mean, args.std))
            probs = logits.softmax(dim=1)
            preds = logits.argmax(dim=1)
            confidence = probs.gather(1, labels.view(-1, 1)).mean().item()
            confidence_meter.update(confidence, batch_size)
            correct += (preds == labels).sum().item()
            total += batch_size
            update_confusion_matrix(confusion, labels.cpu(), preds.cpu())

    print(f"Split: {args.split}")
    print(f"Images: {len(dataset)}")
    print(f"MSE: {mse_meter.avg:.6f}")
    print(f"PSNR: {psnr_meter.avg:.4f}")
    print(f"SSIM: {ssim_meter.avg:.4f}")
    print(f"Recon Loss (BCE): {bce_meter.avg:.6f}")
    print(f"KL Loss: {kl_meter.avg:.6f}")
    print(f"Val Loss (BCE + KL): {bce_meter.avg + kl_meter.avg:.6f}")

    if classifier is not None:
        print(f"Target confidence: {confidence_meter.avg:.4f}")
        print(f"Classifier accuracy on reconstructions: {correct / max(1, total):.4f}")
        print("Confusion matrix:")
        print(format_confusion_matrix(confusion))


if __name__ == "__main__":
    main()
