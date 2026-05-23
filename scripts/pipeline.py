#!/usr/bin/env python3
"""
K-MCFM Film Translation Pipeline (V3 — SDEdit + LoRA)

Usage:
  python scripts/pipeline.py photo.jpg --style portra_400
  python scripts/pipeline.py photo.jpg --style vision3_500t --strength 0.4
  python scripts/pipeline.py ./photos/ --style all --output_dir ./outputs/
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LORAS_DIR = ROOT / "loras"
OUTPUT_DIR = ROOT / "outputs"

FILM_STYLES = {
    "portra_400": {
        "name": "Kodak Portra 400",
        "lora": "portra_400.safetensors",
        "prompt": "a cinematic photograph shot on Kodak Portra 400 film stock, warm natural skin tones, fine grain, natural colors, analog film photography, 35mm",
        "negative_prompt": "digital, oversharpened, HDR, overprocessed, CGI, plastic skin, heavy noise, B&W, anime, illustration",
    },
    "vision3_500t": {
        "name": "Kodak Vision3 500T",
        "lora": "vision3_500t.safetensors",
        "prompt": "a cinematic photograph shot on Kodak Vision3 500T motion picture film, tungsten-balanced, cool cinematic tones, fine grain, analog cinematography, 35mm film still",
        "negative_prompt": "digital, oversharpened, HDR, overprocessed, CGI, plastic skin, heavy noise, B&W, daylight white balance",
    },
    "vision3_250d": {
        "name": "Kodak Vision3 250D",
        "lora": "vision3_250d.safetensors",
        "prompt": "a cinematic photograph shot on Kodak Vision3 250D motion picture film, daylight-balanced, natural colors, fine grain, analog cinematography",
        "negative_prompt": "digital, oversharpened, HDR, overprocessed, CGI, B&W, tungsten white balance",
    },
    "portra_800": {
        "name": "Kodak Portra 800",
        "lora": "portra_800.safetensors",
        "prompt": "a photograph shot on Kodak Portra 800 film, natural skin tones, moderate grain, warm tones, analog film photography, available light",
        "negative_prompt": "digital, oversharpened, HDR, overprocessed, CGI, plastic skin, B&W",
    },
    "ektar_100": {
        "name": "Kodak Ektar 100",
        "lora": "ektar_100.safetensors",
        "prompt": "a vivid photograph shot on Kodak Ektar 100 film, ultra fine grain, rich saturated colors, high contrast, sharp details, analog film photography",
        "negative_prompt": "digital, oversharpened, muted colors, low contrast, B&W",
    },
    "velvia_50": {
        "name": "Fujifilm Velvia 50",
        "lora": "velvia_50.safetensors",
        "prompt": "a photograph shot on Fujifilm Velvia 50 slide film, ultra saturated vivid colors, deep shadows, high contrast, extremely fine grain, landscape photography",
        "negative_prompt": "digital, muted colors, low contrast, B&W, heavy noise, overexposed",
    },
    "hp5": {
        "name": "Ilford HP5 Plus",
        "lora": "hp5.safetensors",
        "prompt": "a black and white photograph shot on Ilford HP5 Plus 400 film, classic grain, rich tonality, wide dynamic range, documentary photography, analog film",
        "negative_prompt": "color, digital, oversharpened, HDR, plastic, CGI, smooth, noise-free",
    },
    "tri_x_400": {
        "name": "Kodak Tri-X 400",
        "lora": "tri_x_400.safetensors",
        "prompt": "a black and white photograph shot on Kodak Tri-X 400 film, classic grain, high contrast, deep shadows, documentary street photography, analog film",
        "negative_prompt": "color, digital, oversharpened, HDR, plastic, CGI, smooth, noise-free",
    },
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    else:
        return torch.device("cpu"), torch.float32


def load_pipeline(device: torch.device, dtype: torch.dtype):
    """Load SDXL img2img pipeline. Downloads model on first run (~7GB)."""
    from diffusers import StableDiffusionXLImg2ImgPipeline

    print(f"Loading SDXL on {device} ({dtype})...")
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    if device.type == "mps":
        pipe = pipe.to(device)
    elif device.type == "cuda":
        pipe = pipe.to(device)

    pipe.safety_checker = None  # Disable for speed
    print(f"  SDXL loaded. VRAM: {torch.mps.current_allocated_memory()//1024//1024 if device.type=='mps' else 'N/A'}MB")
    return pipe


def translate(
    pipe,
    image: Image.Image,
    style: str,
    strength: float = 0.45,
    steps: int = 30,
    guidance: float = 7.5,
    lora_scale: float = 1.0,
) -> Image.Image:
    """SDEdit: noise → denoise with film LoRA and prompt."""
    style_cfg = FILM_STYLES.get(style)
    if not style_cfg:
        raise ValueError(f"Unknown style: {style}. Available: {list(FILM_STYLES)}")

    lora_path = LORAS_DIR / style_cfg["lora"]
    if lora_path.exists():
        print(f"  Loading LoRA: {lora_path.name}")
        pipe.load_lora_weights(str(lora_path))
        # pipe.fuse_lora()  # Speed boost, but quality may vary
    else:
        print(f"  ⚠️  LoRA not found: {lora_path} — using base SDXL only")
        pipe.unload_lora_weights()

    # Resize input to optimal resolution (SDXL works best at 1024²)
    w, h = image.size
    if max(w, h) > 1024:
        scale = 1024 / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        # Ensure divisible by 8
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        image = image.resize((new_w, new_h), Image.LANCZOS)
        print(f"  Resized: {w}x{h} → {new_w}x{new_h}")

    prompt = style_cfg["prompt"]
    neg = style_cfg["negative_prompt"]

    print(f"  Strength: {strength} | Steps: {steps} | CFG: {guidance}")
    print(f"  Prompt: {prompt[:80]}...")

    t0 = time.time()

    result = pipe(
        prompt=prompt,
        negative_prompt=neg,
        image=image,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance,
        cross_attention_kwargs={"scale": lora_scale} if lora_path.exists() else None,
    ).images[0]

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    return result


def main():
    parser = argparse.ArgumentParser(description="K-MCFM Film Translation (SDEdit + LoRA)")
    parser.add_argument("input", type=str, help="Input image or directory")
    parser.add_argument("--style", type=str, default="portra_400",
                        choices=list(FILM_STYLES) + ["all"],
                        help="Film style or 'all'")
    parser.add_argument("--strength", type=float, default=0.45,
                        help="SDEdit strength (0-1). 0.4=subtle, 0.5=moderate, 0.6=strong")
    parser.add_argument("--steps", type=int, default=30,
                        help="Denoising steps. More = quality, less = speed")
    parser.add_argument("--guidance", type=float, default=7.5,
                        help="CFG scale. Higher = more prompt adherence")
    parser.add_argument("--lora-scale", type=float, default=1.0,
                        help="LoRA strength multiplier (0-2)")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory for batch mode")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device, dtype = get_device()
    else:
        device = torch.device(args.device)
        dtype = torch.float16 if args.device == "cuda" else torch.float32

    print(f"=== K-MCFM V3 Film Translation ===")
    print(f"Device: {device}, Input: {args.input}, Style: {args.style}")

    # Load pipeline once
    pipe = load_pipeline(device, dtype)

    try:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"ERROR: {args.input} not found")
            return

        if input_path.is_dir():
            # Batch mode
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            styles = list(FILM_STYLES) if args.style == "all" else [args.style]
            images = sorted([f for f in input_path.iterdir()
                             if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff", ".tif")])

            for img_path in images:
                image = Image.open(img_path).convert("RGB")
                for s in styles:
                    print(f"\n[{img_path.name} → {s}]")
                    result = translate(pipe, image, s,
                                       strength=args.strength, steps=args.steps,
                                       guidance=args.guidance, lora_scale=args.lora_scale)
                    out_name = f"{img_path.stem}_{s}.jpg"
                    out_path = output_dir / out_name
                    result.save(out_path, "JPEG", quality=95)
                    print(f"  Saved: {out_path}")
        else:
            # Single image
            image = Image.open(input_path).convert("RGB")
            styles = list(FILM_STYLES) if args.style == "all" else [args.style]

            for s in styles:
                print(f"\n[→ {s}]")
                result = translate(pipe, image, s,
                                   strength=args.strength, steps=args.steps,
                                   guidance=args.guidance, lora_scale=args.lora_scale)

                if args.output and len(styles) == 1:
                    out_path = Path(args.output)
                else:
                    out_dir = Path(args.output_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{input_path.stem}_{s}.jpg"
                result.save(out_path, "JPEG", quality=95)
                print(f"  Saved: {out_path}")

    finally:
        del pipe
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
