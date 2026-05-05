from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mid_unet_cvae.data import CLASS_TO_INDEX, INDEX_TO_CLASS, MRISliceDataset
from mid_unet_cvae.models import MidUNetCVAE
from mid_unet_cvae.utils import ensure_dir, get_device, load_checkpoint, save_reconstruction_pair, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save same-label reconstruction samples.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--cvae-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/reconstructions")
    parser.add_argument("--split", type=str, choices=["train", "val", "all"], default="val")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def load_cvae(path: str, device: torch.device) -> MidUNetCVAE:
    checkpoint = load_checkpoint(path, device)
    model = MidUNetCVAE(
        image_size=int(checkpoint.get("image_size", 224)),
        latent_dim=int(checkpoint.get("latent_dim", 128)),
        num_classes=int(checkpoint.get("num_classes", len(CLASS_TO_INDEX))),
        class_embed_dim=int(checkpoint.get("class_embed_dim", 32)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    output_dir = ensure_dir(args.output_dir)
    model = load_cvae(args.cvae_checkpoint, device)
    dataset = MRISliceDataset(
        args.data_dir,
        split=args.split,
        image_size=model.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        normalize=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    saved = 0
    for images, labels, subjects, _ in tqdm(loader, desc="reconstruct"):
        images = images.to(device)
        labels = labels.to(device)
        recon = model(images, labels)["recon"]
        class_name = INDEX_TO_CLASS[int(labels.item())].replace(" ", "_")
        filename = f"{saved:04d}_{subjects[0]}_{class_name}.png"
        save_reconstruction_pair(
            original=images[0].cpu(),
            reconstruction=recon[0].cpu(),
            path=output_dir / filename,
            title=f"| {INDEX_TO_CLASS[int(labels.item())]}",
        )
        saved += 1
        if saved >= args.num_samples:
            break

    print(f"Saved {saved} reconstruction samples to {output_dir}")


if __name__ == "__main__":
    main()
