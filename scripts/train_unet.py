#!/usr/bin/env python3
"""
Tiny UNet for content-preserving film color transfer (~500K params).

Architecture forces content preservation:
- 3-level encoder-decoder with skip connections
- Bottleneck is 4×4 spatial → too small for structure, only global color
- Skip connections provide high-frequency detail

Training: color-autoencoder on film images with color corruption.
Model learns to restore film colors from corrupted versions.
At inference, it applies the learned color prior to new images.
"""

import argparse, math, os, random, time, gc
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
LORAS_DIR = ROOT / "loras"

class ImageFolderDataset(Dataset):
    def __init__(self, path, resolution=256):
        self.images = sorted([f for f in Path(path).rglob("*")
                              if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
        self.transform = transforms.Compose([
            transforms.Resize(resolution), transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        try: return self.transform(Image.open(self.images[idx]).convert("RGB"))
        except: return torch.zeros(3, 256, 256)

def color_corrupt(x, severity=0.3):
    """Apply random color distortions to simulate 'digital' look."""
    B, C, H, W = x.shape
    device = x.device

    # Random hue shift
    rgb = x.clone()
    # Random white balance shift
    wb = torch.rand(3, device=device) * severity * 2 + (1 - severity)
    rgb = rgb * wb.view(1, 3, 1, 1)

    # Random gamma shift
    gamma = torch.rand(1, device=device) * severity + (1 - severity/2)
    rgb = rgb.clamp(0, 1) ** gamma.view(1, 1, 1, 1)

    # Random saturation shift
    gray = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    gray = gray.unsqueeze(1)
    sat = torch.rand(1, device=device) * severity * 2 + (1 - severity)
    rgb = gray + (rgb - gray) * sat.unsqueeze(1)

    return rgb.clamp(0, 1)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1), nn.InstanceNorm2d(out_c), nn.ReLU(True),
            nn.Conv2d(out_c, out_c, 3, 1, 1), nn.InstanceNorm2d(out_c), nn.ReLU(True),
        )
    def forward(self, x): return self.body(x)

class TinyFilmUNet(nn.Module):
    def __init__(self, base_c=16):
        super().__init__()
        self.enc1 = ConvBlock(3, base_c)
        self.enc2 = ConvBlock(base_c, base_c*2)
        self.enc3 = ConvBlock(base_c*2, base_c*4)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_c*4, base_c*4, 3, 1, 1),
            nn.InstanceNorm2d(base_c*4), nn.ReLU(True),
        )

        self.up3 = nn.ConvTranspose2d(base_c*4, base_c*4, 2, 2)
        self.dec3 = ConvBlock(base_c*8, base_c*2)  # 64+64=128

        self.up2 = nn.ConvTranspose2d(base_c*2, base_c*2, 2, 2)
        self.dec2 = ConvBlock(base_c*4, base_c)    # 32+32=64

        self.up1 = nn.ConvTranspose2d(base_c, base_c, 2, 2)
        self.dec1 = ConvBlock(base_c*2, base_c)     # 16+16=32

        self.final = nn.Sequential(nn.Conv2d(base_c, 3, 3, 1, 1), nn.Tanh())
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x_norm):
        e1 = self.enc1(x_norm)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--film-dir", required=True, help="Film domain images")
    parser.add_argument("--style", default="film", help="Style name")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = TinyFilmUNet(base_c=16).to(device)
    print(f"=== TinyFilmUNet {args.style}: {sum(p.numel() for p in model.parameters()):,} params ===")
    print(f"Film dir: {args.film_dir}")

    ds = ImageFolderDataset(args.film_dir, args.resolution)
    print(f"Film images: {len(ds)}")
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    # VGG for perceptual loss
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
    for p in vgg.parameters(): p.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total = 0
        for film in dl:
            film = film.to(device)

            # Corrupted version (simulates digital input)
            corrupted = color_corrupt(film.clone(), severity=0.25 + random.random() * 0.2)

            # Normalize both to [-1, 1]
            film_norm = film * 2 - 1
            corr_norm = corrupted * 2 - 1

            output = model(corr_norm)

            loss_l1 = F.l1_loss(output, film_norm)
            loss_p = F.l1_loss(vgg((output+1)/2), vgg((film_norm+1)/2))
            loss = loss_l1 + 0.05 * loss_p

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        sched.step()
        avg = total / len(dl)
        print(f"  {epoch+1}/{args.epochs}: loss={avg:.4f}")

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), LORAS_DIR / f"{args.style}_unet.pth")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), LORAS_DIR / f"{args.style}_unet.pth")
        torch.mps.empty_cache()

    print(f"\nBest loss: {best_loss:.4f}")
    torch.save(model.state_dict(), LORAS_DIR / f"{args.style}_unet.pth")

    # ── Inference ──
    print("Inference...")
    model.eval()
    img = Image.open("outputs/original.jpg").convert("RGB")
    t = transforms.Compose([transforms.Resize(args.resolution), transforms.CenterCrop(args.resolution), transforms.ToTensor()])
    x = t(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x * 2 - 1)
        out = ((out + 1) / 2).clamp(0, 1).cpu().squeeze(0).permute(1,2,0).numpy()
        out = (out * 255).astype(np.uint8)
    Image.fromarray(out).save(f"outputs/{args.style}_unet.jpg", "JPEG", quality=95)

    # Comparison
    img_small = img.resize((args.resolution, args.resolution))
    comp = Image.new("RGB", (args.resolution * 2, args.resolution))
    comp.paste(img_small, (0, 0))
    comp.paste(Image.fromarray(out), (args.resolution, 0))
    comp.save(f"outputs/comparison_{args.style}.jpg", "JPEG", quality=95)
    print(f"  outputs/{args.style}_unet.jpg + comparison")


if __name__ == "__main__":
    main()
