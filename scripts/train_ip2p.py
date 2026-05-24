#!/usr/bin/env python3
"""
Fine-tune InstructPix2Pix on custom film paired data (M5 optimized).

Creates paired data from film scans via color perturbation,
then fine-tunes IP2P to map digital→film colors.

Usage:
  python scripts/train_ip2p.py --film-dir data/film_domain/portra_400 --style portra_400 --steps 500
"""

import argparse, json, random, time, gc
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
LORAS_DIR = ROOT / "loras"

class FilmPairDataset(Dataset):
    """Generate pseudo-pairs: film image + color-corrupted 'digital' version."""
    def __init__(self, film_dir, resolution=256, size=200):
        self.images = sorted([f for f in Path(film_dir).rglob("*")
                              if f.suffix.lower() in (".jpg", ".jpeg", ".png")])[:size]
        self.transform = transforms.Compose([
            transforms.Resize(resolution), transforms.CenterCrop(resolution),
            transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        self.resolution = resolution

    def __len__(self): return len(self.images)

    def corrupt_colors(self, img_np):
        """Simulate digital camera color profile."""
        img = img_np.astype(np.float32)
        # Random white balance
        wb = np.array([0.7 + random.random() * 0.6, 0.7 + random.random() * 0.6, 0.7 + random.random() * 0.6])
        img = img * wb
        # Random gamma
        gamma = 0.6 + random.random() * 0.8
        img = np.clip(255 * (np.clip(img / 255.0, 0, 1) ** gamma), 0, 255)
        return img.astype(np.uint8)

    def __getitem__(self, idx):
        film = Image.open(self.images[idx]).convert("RGB")
        film_np = np.array(film)
        digital = Image.fromarray(self.corrupt_colors(film_np))
        return self.transform(digital), self.transform(film)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--film-dir", required=True)
    parser.add_argument("--style", default="portra_400")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32  # fp32 for MPS training stability

    print(f"=== IP2P Fine-tune: {args.style} ===")
    print(f"Film: {args.film_dir}, Steps: {args.steps}, LR: {args.lr}")

    from diffusers import StableDiffusionInstructPix2PixPipeline, DDPMScheduler

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=dtype, safety_checker=None).to(device)
    pipe.scheduler = DDPMScheduler.from_pretrained("timbrooks/instruct-pix2pix", subfolder="scheduler")

    unet, vae, te, tok = pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer
    noise_scheduler = pipe.scheduler

    # Enable training
    unet.train()
    unet.enable_gradient_checkpointing()
    for p in vae.parameters(): p.requires_grad = False
    for p in te.parameters(): p.requires_grad = False

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"  UNet params: {trainable:,}")

    ds = FilmPairDataset(args.film_dir, args.resolution, size=200)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    print(f"  Training pairs: {len(ds)}")

    prompt = f"apply {args.style.replace('_', ' ').title()} film color grading, warm natural tones, soft contrast, fine grain, analog film look"
    with torch.no_grad():
        tokens = tok(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        eh = te(tokens["input_ids"].to(device))[0]

    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    sched = CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.1)

    best_loss = float("inf")
    progress = tqdm(total=args.steps, desc="Training")
    data_iter = iter(dl)
    global_step = 0

    while global_step < args.steps:
        try: digital, film = next(data_iter)
        except StopIteration: data_iter = iter(dl); digital, film = next(data_iter)

        digital, film = digital.to(device, dtype=dtype), film.to(device, dtype=dtype)

        with torch.no_grad():
            film_latent = vae.encode(film).latent_dist.sample() * vae.config.scaling_factor
            digital_latent = vae.encode(digital).latent_dist.sample() * vae.config.scaling_factor

        noise = torch.randn_like(film_latent, dtype=dtype)
        timesteps = torch.randint(0, 1000, (1,), device=device).long()
        noisy_latent = noise_scheduler.add_noise(film_latent, noise, timesteps)
        unet_input = torch.cat([noisy_latent, digital_latent], dim=1)

        noise_pred = unet(unet_input, timesteps, encoder_hidden_states=eh).sample
        loss = F.mse_loss(noise_pred, noise)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        global_step += 1

        loss_val = loss.item()
        progress.update(1)
        progress.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{sched.get_last_lr()[0]:.1e}"})

        if loss_val < best_loss:
            best_loss = loss_val
        if global_step % args.save_every == 0 or global_step >= args.steps:
            out = LORAS_DIR / f"ip2p_{args.style}"
            pipe.save_pretrained(str(out))
            tqdm.write(f"  Saved: {out}")

        torch.mps.empty_cache()

    progress.close()
    print(f"\nBest loss: {best_loss:.4f}")
    print(f"Model: {LORAS_DIR / f'ip2p_{args.style}'}")


if __name__ == "__main__":
    main()
