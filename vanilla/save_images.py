import sys
from pathlib import Path
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# src 폴더 경로 추가
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from vanilla_cvae.data import MRISliceDataset
from vanilla_cvae.utils import load_checkpoint, get_device
from vanilla_cvae.models import VanillaCVAE

def main():
    device = get_device()
    checkpoint_path = "checkpoints/vanilla_cvae/best_vanilla_cvae.pth"
    data_dir = "data/Data"
    
    print("모델과 데이터를 불러오는 중입니다...")
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = VanillaCVAE(
        img_size=224, channels=1, num_classes=3, latent_dim=128
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = MRISliceDataset(data_dir, split="val", image_size=224, normalize=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    images, labels, _, _ = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)
    
    print("이미지를 복원하는 중입니다...")
    with torch.no_grad():
        output = model(images, labels)
        recon = output["recon"]
        
    # 윗줄: 원본(Original), 아랫줄: 복원(Reconstruction)
    comparison = torch.cat([images, recon])
    save_path = "vanilla_reconstruction_results.png"
    save_image(comparison, save_path, nrow=8)
    print(f"이미지 저장이 완료되었습니다! 파일명: {save_path}")

if __name__ == "__main__":
    main()
