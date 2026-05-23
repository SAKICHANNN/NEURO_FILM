#!/usr/bin/env python3
"""Train SDXL LoRA for a film stock using real film scans.

Usage:
  python scripts/train_lora.py --data data/film_domain/velvia_50 --name velvia_50
  python scripts/train_lora.py --data data/film_domain/hp5 --name hp5 --rank 8 --steps 1500
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
LORAS_DIR = ROOT / "loras"

FILM_CAPTIONS = {
    "velvia_50": "a photograph shot on Fujifilm Velvia 50 slide film, ultra saturated vivid colors, deep shadows, extremely fine grain, landscape photography, analog film",
    "hp5": "a black and white photograph shot on Ilford HP5 Plus 400 film, classic grain, rich tonality, wide dynamic range, documentary photography, analog film",
    "portra_400": "a photograph shot on Kodak Portra 400 film, warm natural skin tones, fine grain, analog photography",
    "portra_800": "a photograph shot on Kodak Portra 800 film, natural tones, moderate grain, warm colors, analog photography",
    "vision3_500t": "a cinematic photograph shot on Kodak Vision3 500T film, tungsten tones, film grain, analog cinematography",
    "vision3_250d": "a cinematic photograph shot on Kodak Vision3 250D film, daylight colors, fine grain, analog cinematography",
    "ektar_100": "a vivid photograph shot on Kodak Ektar 100 film, ultra fine grain, rich saturated colors, analog photography",
    "tri_x_400": "a black and white photograph shot on Kodak Tri-X 400 film, classic grain, high contrast, documentary photography",
}


class FilmDataset(Dataset):
    def __init__(self, image_dir: Path, caption: str, resolution: int = 1024):
        self.image_dir = image_dir
        self.caption = caption
        self.resolution = resolution
        self.images = sorted([
            f for f in image_dir.rglob("*")
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert("RGB")
            return self.transform(img), self.caption
        except Exception:
            return torch.zeros(3, self.resolution, self.resolution), self.caption


def train_lora(
    data_dir: Path,
    output_name: str,
    caption: str,
    resolution: int = 768,
    rank: int = 16,
    steps: int = 1500,
    batch_size: int = 1,
    gradient_accumulation: int = 4,
    lr: float = 1e-4,
    save_every: int = 500,
    device: str = "mps",
):
    # ── Check data ──
    images = sorted([
        f for f in data_dir.rglob("*")
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])
    if len(images) < 50:
        print(f"ERROR: Only {len(images)} images. Need at least 50.")
        return
    print(f"Dataset: {len(images)} images")

    # ── Device ──
    weight_dtype = torch.float32  # MPS fp16 training is unstable for SDXL
    device = torch.device(device)

    # ── Load base model ──
    print("Loading SDXL base model...")
    from diffusers import StableDiffusionXLPipeline
    from diffusers.loaders import LoraLoaderMixin

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=weight_dtype,
        use_safetensors=True,
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    text_encoder = pipeline.text_encoder
    text_encoder_2 = pipeline.text_encoder_2
    tokenizer = pipeline.tokenizer
    tokenizer_2 = pipeline.tokenizer_2
    vae = pipeline.vae
    unet = pipeline.unet

    # Freeze base models
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # ── Add LoRA layers ──
    print(f"Adding LoRA layers (rank={rank})...")
    from peft import LoraConfig, get_peft_model

    target_modules = [
        "to_q", "to_k", "to_v", "to_out.0",
        "add_k_proj", "add_v_proj", "add_q_proj", "add_out_proj.0",
    ]

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )

    unet_lora = get_peft_model(unet, lora_config)
    trainable_params = sum(p.numel() for p in unet_lora.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable_params:,} (~{trainable_params * 2 // 1024 // 1024}MB)")

    # ── Data ──
    dataset = FilmDataset(data_dir, caption, resolution)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        [p for p in unet_lora.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.1)

    # ── Noise scheduler ──
    from diffusers import DDPMScheduler
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # ── Text encoding ──
    print("Encoding prompt...")
    with torch.no_grad():
        text_inputs = tokenizer(caption, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        text_inputs_2 = tokenizer_2(caption, padding="max_length", max_length=77, truncation=True, return_tensors="pt")

        # Encode with both text encoders
        # text_encoder (CLIP-L): [0]=last_hidden_state (1,77,768)
        # text_encoder_2 (OpenCLIP-G): [1] or .last_hidden_state (1,77,1280), .text_embeds (1,1280)
        prompt_embeds = text_encoder(text_inputs["input_ids"].to(device))[0]
        # text_encoder_2: call once, get both last_hidden_state and text_embeds
        text_enc2_output = text_encoder_2(text_inputs_2["input_ids"].to(device), output_hidden_states=True)
        prompt_embeds_2_seq = text_enc2_output.last_hidden_state  # (1,77,1280)
        prompt_embeds_2_pooled = text_enc2_output.text_embeds      # (1,1280)

        # SDXL UNet expects concatenated: cat([CLIP-L, OpenCLIP-G], dim=-1) = (1,77,2048)
        encoder_hidden_states = torch.cat([prompt_embeds, prompt_embeds_2_seq], dim=-1)

        # pooled embeds for added_cond_kwargs
        add_text_embeds = prompt_embeds_2_pooled
        add_text_embeds = add_text_embeds / add_text_embeds.norm(dim=-1, keepdim=True) * 0.13025

    # ── Training loop ──
    print(f"\nTraining {steps} steps (lr={lr}, accum={gradient_accumulation}, res={resolution}²)...")
    print(f"Output: {LORAS_DIR / f'{output_name}.safetensors'}")

    unet_lora.train()
    global_step = 0
    epoch = 0
    best_loss = float("inf")

    progress = tqdm(total=steps, desc="Training")
    data_iter = iter(dataloader)

    while global_step < steps:
        epoch += 1

        for micro_step in range(gradient_accumulation):
            try:
                images_batch, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                images_batch, _ = next(data_iter)

            images_batch = images_batch.to(device, dtype=weight_dtype)

            with torch.no_grad():
                latents = vae.encode(images_batch).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            bsz = latents.shape[0]

            # SDXL needs additional args
            # ── SDXL time_ids: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
            time_ids = torch.tensor(
                [[resolution, resolution, 0, 0, resolution, resolution]],
                device=device, dtype=weight_dtype
            ).repeat(bsz, 1)

            # Predict noise with LoRA UNet
            with torch.amp.autocast(device_type=device.type):
                noise_pred = unet_lora(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states.repeat(bsz, 1, 1),
                added_cond_kwargs={
                    "text_embeds": add_text_embeds.repeat(bsz, 1),
                    "time_ids": time_ids,
                },
            ).sample

            loss = F.mse_loss(noise_pred, noise, reduction="mean")
            loss = loss / gradient_accumulation

            loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        global_step += 1
        loss_val = loss.item() * gradient_accumulation

        progress.update(1)
        progress.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        if global_step % save_every == 0 or global_step >= steps:
            output_path = LORAS_DIR / output_name
            output_path.mkdir(parents=True, exist_ok=True)

            # Extract LoRA weights and save
            from safetensors.torch import save_file
            lora_state = {}
            for name, param in unet_lora.named_parameters():
                if param.requires_grad and "lora" in name:
                    safe_name = name.replace(".", "_").replace("base_model_model_", "")
                    lora_state[safe_name] = param.detach().cpu()

            checkpoint_path = output_path / f"{output_name}_step{global_step}.safetensors"
            save_file(lora_state, str(checkpoint_path))
            tqdm.write(f"  Saved: {checkpoint_path}")

            if loss_val < best_loss:
                best_loss = loss_val
                final_path = LORAS_DIR / f"{output_name}.safetensors"
                import shutil
                shutil.copy(checkpoint_path, final_path)
                tqdm.write(f"  Best model saved: {final_path} (loss={loss_val:.4f})")

    progress.close()
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"LoRA saved to: {LORAS_DIR / f'{output_name}.safetensors'}")


def main():
    parser = argparse.ArgumentParser(description="Train SDXL LoRA for film stock")
    parser.add_argument("--data", type=str, required=True, help="Image directory")
    parser.add_argument("--name", type=str, required=True, help="Output LoRA name")
    parser.add_argument("--caption", type=str, help="Training caption (default: auto)")
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"ERROR: {args.data} not found")
        return

    caption = args.caption or FILM_CAPTIONS.get(
        args.name, f"a photograph shot on {args.name} film, analog photography"
    )

    print(f"=== Training SDXL LoRA: {args.name} ===")
    print(f"Data: {data_dir}")
    print(f"Caption: {caption}")

    train_lora(
        data_dir=data_dir,
        output_name=args.name,
        caption=caption,
        resolution=args.resolution,
        rank=args.rank,
        steps=args.steps,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
