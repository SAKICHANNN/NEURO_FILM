#!/usr/bin/env python3
"""Fine-tune SD 1.5 cross-attention layers for film style (MPS-compatible).

Selective fine-tuning of UNet cross-attention to_k/to_v layers.
Extracts weight diffs as compact LoRA after training.

Usage:
  python scripts/train_lora_sd.py --data data/film_domain/velvia_50 --name velvia_50 --steps 1000
"""

import argparse, copy, gc, math, os, time
from pathlib import Path
import torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
LORAS_DIR = ROOT / "loras"

CAPTIONS = {
    "velvia_50": "Fujifilm Velvia 50 slide film photograph, saturated colors, fine grain",
    "hp5": "Ilford HP5 Plus 400 black and white film photograph, classic grain",
}

class FilmDataset(Dataset):
    def __init__(self, image_dir, caption, resolution=512):
        self.images = sorted([f for f in Path(image_dir).rglob("*") if f.suffix.lower() in (".jpg",".jpeg",".png")])
        self.caption = caption
        self.transform = transforms.Compose([
            transforms.Resize(resolution, transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution), transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        try: return self.transform(Image.open(self.images[idx]).convert("RGB")), self.caption
        except: return torch.zeros(3,512,512), self.caption


def save_lora_diff(original_state, finetuned_state, output_path, rank=16):
    """Extract LoRA from weight differences using SVD."""
    from safetensors.torch import save_file
    lora_weights = {}
    for name in original_state:
        if name not in finetuned_state: continue
        if 'attn2.to_k' not in name and 'attn2.to_v' not in name: continue
        diff = finetuned_state[name] - original_state[name].cpu()
        if diff.abs().max() < 1e-6: continue
        U, S, V = torch.svd_lowrank(diff.float(), q=rank)
        safe_name = name.replace(".", "_")
        lora_weights[f"lora_A_{safe_name}"] = (U * S.sqrt()).half()
        lora_weights[f"lora_B_{safe_name}"] = (V * S.sqrt()).half()
    save_file(lora_weights, str(output_path))
    print(f"  LoRA saved: {output_path} ({len(lora_weights)} tensors)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--caption", default=None)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=250)
    args = parser.parse_args()

    device = torch.device("mps")
    dtype = torch.float32
    caption = args.caption or CAPTIONS.get(args.name, f"{args.name} film photograph")

    print(f"=== SD 1.5 Fine-tune: {args.name} ===")
    print(f"Res: {args.resolution}², Steps: {args.steps}, LR: {args.lr}")

    # ── Load model ──
    print("Loading SD 1.5...")
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=dtype, use_safetensors=True,
        safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    unet, vae, text_encoder, tokenizer = pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer

    # Save original weights for LoRA extraction later
    original_state = {k: v.cpu().clone() for k, v in unet.state_dict().items() if 'attn2.to_k' in k or 'attn2.to_v' in k}

    # Freeze all, then unfreeze cross-attention to_k/to_v
    for p in unet.parameters(): p.requires_grad = False
    for name, p in unet.named_parameters():
        if 'attn2.to_k' in name or 'attn2.to_v' in name:
            p.requires_grad = True

    unet.train()
    for m in [vae, text_encoder]: m.eval()

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,} params (~{trainable*4//1024//1024}MB)")

    # ── Data ──
    images = sorted([f for f in Path(args.data).rglob("*") if f.suffix.lower() in (".jpg",".jpeg",".png")])
    print(f"  Images: {len(images)}")
    ds = FilmDataset(args.data, caption, args.resolution)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    # ── Encode prompt ──
    with torch.no_grad():
        tokens = tokenizer(caption, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        eh = text_encoder(tokens["input_ids"].to(device))[0]

    # ── Noise scheduler ──
    from diffusers import DDPMScheduler
    noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    # ── Optimizer ──
    trainable_list = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_list, lr=args.lr)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr*0.1)

    # ── Training ──
    print(f"\nTraining...")
    out_path = LORAS_DIR / f"{args.name}_sd15.safetensors"
    progress = tqdm(total=args.steps, desc="Training")
    global_step, best_loss = 0, float("inf")
    data_iter = iter(dl)

    while global_step < args.steps:
        try: images_batch, _ = next(data_iter)
        except StopIteration: data_iter = iter(dl); images_batch, _ = next(data_iter)
        images_batch = images_batch.to(device, dtype=dtype)

        with torch.no_grad():
            latents = vae.encode(images_batch).latent_dist.sample() * vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=eh).sample
        loss = F.mse_loss(noise_pred, noise)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_list, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

        loss_val = loss.item()
        progress.update(1)
        progress.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.1e}"})

        if global_step % args.save_every == 0 or global_step >= args.steps:
            finetuned_state = {k: v.cpu().clone() for k, v in unet.state_dict().items() if 'attn2.to_k' in k or 'attn2.to_v' in k}
            save_lora_diff(original_state, finetuned_state, out_path)
            if loss_val < best_loss:
                best_loss = loss_val
            torch.mps.empty_cache()

    progress.close()
    print(f"\nDone! Best loss: {best_loss:.4f}")
    print(f"LoRA: {out_path}")


if __name__ == "__main__":
    main()
