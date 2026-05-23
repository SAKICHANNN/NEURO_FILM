#!/usr/bin/env python3
"""
K-MCFM Film Translation — SD 1.5 + Self-Trained LoRAs (M5 optimized)

8 film stocks, all trained on real Flickr film scans.
Merge LoRA into UNet at runtime, single-pass img2img inference.

Usage:
  python scripts/translate.py photo.jpg --style velvia_50
  python scripts/translate.py ./photos/ --style all
"""

import argparse, gc, sys, time
from pathlib import Path
from collections import defaultdict
import torch
from PIL import Image
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
LORAS_DIR = ROOT / "loras"
OUTPUT_DIR = ROOT / "outputs"

STYLES = {
    "velvia_50":     "Fujifilm Velvia 50 slide film, ultra saturated vivid colors, deep shadows, extremely fine grain, analog film",
    "hp5":           "Ilford HP5 Plus 400 black and white film, classic grain, rich tonality, documentary photography, analog film",
    "portra_400":    "Kodak Portra 400 film, warm natural skin tones, fine grain, soft contrast, analog film photography",
    "portra_800":    "Kodak Portra 800 film, natural skin tones, moderate grain, warm colors, analog film photography",
    "vision3_500t":  "Kodak Vision3 500T tungsten film, cinematic cool tones, film grain, motion picture film",
    "vision3_250d":  "Kodak Vision3 250D daylight film, natural colors, fine grain, motion picture film",
    "ektar_100":     "Kodak Ektar 100 film, ultra fine grain, rich saturated colors, high contrast, analog film",
    "tri_x_400":     "Kodak Tri-X 400 black and white film, classic grain, high contrast, deep shadows, documentary photography",
}

NEG_PROMPT = "digital, oversharpened, HDR, CGI, plastic, smooth, noise-free"


def merge_lora(unet, lora_path: Path, scale: float = 0.6):
    """Merge LoRA weight diffs into UNet state dict, return original state for restore."""
    pairs = defaultdict(dict)
    f = safe_open(lora_path, framework="pt")
    for k in f.keys():
        parts = k.split("_", 2)
        pairs[parts[2][:-7]][parts[1]] = f.get_tensor(k)

    state = unet.state_dict()
    original = {}
    for base, mats in pairs.items():
        unet_key = base.replace("_", ".") + ".weight"
        if unet_key not in state:
            continue
        A = mats["A"].to(state[unet_key].device, state[unet_key].dtype)
        B = mats["B"].to(state[unet_key].device, state[unet_key].dtype)
        diff = (A @ B.T) * scale
        original[unet_key] = state[unet_key].clone()
        state[unet_key] = state[unet_key] + diff
    unet.load_state_dict(state)
    return original


def restore_unet(unet, original):
    state = unet.state_dict()
    state.update(original)
    unet.load_state_dict(state)


def main():
    parser = argparse.ArgumentParser(description="K-MCFM Film Translation")
    parser.add_argument("input", help="Image or directory")
    parser.add_argument("--style", default="velvia_50", choices=list(STYLES) + ["all"])
    parser.add_argument("--strength", type=float, default=0.45)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", help="Output file")
    args = parser.parse_args()

    device = torch.device("mps")
    dtype = torch.float16

    print("Loading SD 1.5...")
    t0 = time.time()
    from diffusers import StableDiffusionImg2ImgPipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype, use_safetensors=True, safety_checker=None,
    ).to(device)
    print(f"  {time.time()-t0:.0f}s")

    input_path = Path(args.input)
    images = []
    if input_path.is_dir():
        images = sorted([f for f in input_path.iterdir() if f.suffix.lower() in (".jpg",".jpeg",".png",".tiff",".tif")])
    elif input_path.exists():
        images = [input_path]
    else:
        print(f"Not found: {args.input}"); return

    styles = list(STYLES) if args.style == "all" else [args.style]

    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        for style in styles:
            lora_path = LORAS_DIR / f"{style}_sd15.safetensors"
            if not lora_path.exists():
                print(f"[{style}] No LoRA found, skipping"); continue

            print(f"\n[{img_path.name} → {style}]")
            t0 = time.time()
            original = merge_lora(pipe.unet, lora_path)
            merge_time = time.time() - t0

            result = pipe(
                prompt=STYLES[style],
                negative_prompt=NEG_PROMPT,
                image=image, strength=args.strength,
                num_inference_steps=args.steps, guidance_scale=7.5,
            ).images[0]

            restore_unet(pipe.unet, original)
            elapsed = time.time() - t0

            out_dir = OUTPUT_DIR if img_path.is_dir() or args.style == "all" else Path(".")
            out_dir.mkdir(parents=True, exist_ok=True)
            if args.output and len(styles) == 1:
                out_path = Path(args.output)
            else:
                out_path = out_dir / f"{img_path.stem}_{style}.jpg"
            result.save(out_path, "JPEG", quality=92)
            print(f"  {elapsed:.1f}s (merge {merge_time:.1f}s) → {out_path}")

    del pipe; gc.collect(); torch.mps.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
