#!/usr/bin/env python3
"""
Content-Preserving Film Color Transfer via 3D LUT Prediction

Architecture: ResNet-18 predicts 3D LUT blending weights.
LUT application is deterministic per-pixel → content guaranteed preserved.
Trained with unpaired GAN (PatchGAN discriminator judges "film look").

Based on: Image-Adaptive-3DLUT (Zeng et al., TPAMI 2021)
"""

import sys, os, time, argparse, copy, random, gc
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
LORAS_DIR = ROOT / "loras"

# ── 3D LUT with trilinear interpolation (deterministic → content preserved) ──
class TrilinearLUT(nn.Module):
    """Differentiable 3D LUT. Input (B,3,H,W) in [0,1] → output (B,3,H,W) in [0,1]."""
    def __init__(self, dim=33):
        super().__init__()
        self.dim = dim
        # Initialize as identity LUT
        base = torch.linspace(0, 1, dim)
        r, g, b = torch.meshgrid(base, base, base, indexing='ij')
        self.lut = nn.Parameter(torch.stack([r, g, b], dim=-1).permute(3,0,1,2).unsqueeze(0))  # (1,3,D,D,D)

    def forward(self, x):
        # x: (B,3,H,W) in [0,1]
        # LUT: (1,3,D,D,D)
        x = x.permute(0, 2, 3, 1).contiguous()  # (B,H,W,3)
        x_clamped = x.clamp(0, 1)
        idx = x_clamped * (self.dim - 1)
        idx0 = idx.floor().long().clamp(0, self.dim - 2)
        idx1 = idx0 + 1
        d = idx - idx0.float()

        # Trilinear interpolation
        lut = self.lut[0].permute(1,2,3,0)  # (D,D,D,3)
        c000 = lut[idx0[...,0], idx0[...,1], idx0[...,2]]
        c001 = lut[idx0[...,0], idx0[...,1], idx1[...,2]]
        c010 = lut[idx0[...,0], idx1[...,1], idx0[...,2]]
        c011 = lut[idx0[...,0], idx1[...,1], idx1[...,2]]
        c100 = lut[idx1[...,0], idx0[...,1], idx0[...,2]]
        c101 = lut[idx1[...,0], idx0[...,1], idx1[...,2]]
        c110 = lut[idx1[...,0], idx1[...,1], idx0[...,2]]
        c111 = lut[idx1[...,0], idx1[...,1], idx1[...,2]]

        dx, dy, dz = d[...,0:1], d[...,1:2], d[...,2:3]
        c00 = c000 * (1 - dx) + c100 * dx
        c01 = c001 * (1 - dx) + c101 * dx
        c10 = c010 * (1 - dx) + c110 * dx
        c11 = c011 * (1 - dx) + c111 * dx
        c0 = c00 * (1 - dy) + c10 * dy
        c1 = c01 * (1 - dy) + c11 * dy
        c = c0 * (1 - dz) + c1 * dz

        return c.permute(0, 3, 1, 2).contiguous()


# ── Generator: ResNet-18 predicts LUT blending weights ──
class LUTGenerator(nn.Module):
    def __init__(self, n_luts=3, n_weights=3):
        super().__init__()
        net = models.resnet18(weights=None)
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear')
        net.fc = nn.Linear(512, n_weights)
        self.model = net

    def forward(self, x):
        x = self.upsample(x)
        return self.model(x)


# ── Full model: Generator + multiple basis LUTs ──
class AdaptiveLUTModel(nn.Module):
    def __init__(self, n_luts=3):
        super().__init__()
        self.generator = LUTGenerator(n_luts=n_luts, n_weights=n_luts)
        self.luts = nn.ModuleList([TrilinearLUT(dim=17) for _ in range(n_luts)])

    def forward(self, x):
        weights = self.generator(x)  # (B, n_luts)
        weights = F.softmax(weights, dim=1)
        out = 0
        for i in range(len(self.luts)):
            w = weights[:, i:i+1].unsqueeze(-1).unsqueeze(-1)  # (B,1,1,1)
            out = out + w * self.luts[i](x)
        return out


# ── PatchGAN Discriminator ──
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        def conv_block(in_f, out_f, norm=True):
            layers = [nn.Conv2d(in_f, out_f, 4, 2, 1)]
            if norm: layers.append(nn.InstanceNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *conv_block(in_channels, 64, norm=False),
            *conv_block(64, 128),
            *conv_block(128, 256),
            *conv_block(256, 512),
            nn.Conv2d(512, 1, 4, 2, 1),
        )

    def forward(self, x):
        return self.model(x)


# ── Dataset ──
class ImageFolderDataset(Dataset):
    def __init__(self, path, resolution=256):
        self.images = sorted([f for f in Path(path).rglob("*")
                              if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])

    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        try: return self.transform(Image.open(self.images[idx]).convert("RGB"))
        except: return torch.zeros(3, 256, 256)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="outputs", help="Digital images dir")
    parser.add_argument("--target", required=True, help="Film domain images dir")
    parser.add_argument("--style", default="portra_400", help="Style name")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32

    print(f"=== Film Color Transfer: {args.style} ===")
    print(f"Source: {args.source}, Target: {args.target}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")

    # ── Models ──
    model = AdaptiveLUTModel(n_luts=3).to(device)
    discriminator = PatchGANDiscriminator().to(device)

    # Try loading pretrained FiveK weights
    pretrained_dir = Path("/tmp/3dlut/pretrained_models/sRGB")
    if pretrained_dir.exists():
        try:
            model.generator.load_state_dict(
                torch.load(pretrained_dir / "classifier_unpaired.pth", map_location=device), strict=False)
            print("Loaded pretrained generator (FiveK)")
        except: pass

    # ── Data ──
    source_ds = ImageFolderDataset(args.source, args.resolution)
    target_ds = ImageFolderDataset(args.target, args.resolution)
    print(f"Source: {len(source_ds)}, Target: {len(target_ds)} images")

    source_dl = DataLoader(source_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    target_dl = DataLoader(target_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # ── Optimizers ──
    opt_G = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # ── Losses ──
    criterion_gan = nn.MSELoss()
    criterion_identity = nn.L1Loss()
    criterion_color = nn.L1Loss()

    # Perceptual loss via VGG (color histogram matching)
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to(device)
    except:
        lpips_fn = None

    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        discriminator.train()

        for i, (real_A, real_B) in enumerate(zip(source_dl, target_dl)):
            real_A = real_A.to(device)  # digital
            real_B = real_B.to(device)  # film

            # ── Train Generator ──
            opt_G.zero_grad()
            fake_B = model(real_A)

            # GAN loss: fool discriminator
            pred_fake = discriminator(fake_B)
            loss_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))

            # Identity loss: preserve content
            loss_idt = criterion_identity(fake_B, real_A) * 5.0

            # Color histogram loss: match film color distribution
            loss_color = 0
            for c in range(3):
                hist_fake = torch.histc(fake_B[:, c], bins=64, min=0, max=1)
                hist_real = torch.histc(real_B[:, c], bins=64, min=0, max=1)
                loss_color += criterion_color(hist_fake / hist_fake.sum(),
                                              hist_real / hist_real.sum())
            loss_color = loss_color * 3.0

            loss_G = loss_gan + loss_idt + loss_color
            loss_G.backward()
            opt_G.step()

            # ── Train Discriminator ──
            opt_D.zero_grad()
            pred_real = discriminator(real_B)
            loss_D_real = criterion_gan(pred_real, torch.ones_like(pred_real))
            pred_fake = discriminator(fake_B.detach())
            loss_D_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            if i % 50 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs} | Step {i} | "
                      f"G={loss_G.item():.3f} D={loss_D.item():.3f} "
                      f"color={loss_color.item():.4f}")

        # ── Save checkpoint ──
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            ckpt = LORAS_DIR / f"{args.style}_lut.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"  Saved: {ckpt}")

        torch.mps.empty_cache()

    # ── Final save ──
    final_path = LORAS_DIR / f"{args.style}_lut.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete: {final_path}")


if __name__ == "__main__":
    main()
