#!/usr/bin/env python3
"""Merge our self-trained LoRA into SD 1.5 UNet + run inference."""

import sys, time, torch
from pathlib import Path
from PIL import Image
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
LORAS_DIR = ROOT / "loras"

CAPTIONS = {
    "velvia_50": "Fujifilm Velvia 50 slide film, saturated colors, fine grain",
    "hp5": "Ilford HP5 Plus 400 black and white film, classic grain",
    "portra_400": "Kodak Portra 400 film, warm natural skin tones, fine grain",
    "portra_800": "Kodak Portra 800 film, moderate grain, warm tones",
    "vision3_500t": "Kodak Vision3 500T tungsten film, cinematic tones",
    "vision3_250d": "Kodak Vision3 250D daylight film, natural colors",
    "ektar_100": "Kodak Ektar 100 film, ultra fine grain, saturated",
    "tri_x_400": "Kodak Tri-X 400 black and white film, classic grain, high contrast",
}


def merge_lora(unet, lora_path):
    """Merge LoRA weight diffs into UNet in-place."""
    from collections import defaultdict
    f = safe_open(lora_path, framework="pt")
    pairs = defaultdict(dict)
    for k in f.keys():
        parts = k.split("_", 2)
        pairs[parts[2][:-7]][parts[1]] = f.get_tensor(k)

    state = unet.state_dict()
    for base, mats in pairs.items():
        unet_key = base.replace("_", ".") + ".weight"
        if unet_key not in state:
            continue
        A = mats["A"]  # (in_dim, rank)
        B = mats["B"]  # (out_dim, rank)
        diff = (A @ B.T).to(state[unet_key].dtype).to(state[unet_key].device)
        state[unet_key] = state[unet_key] + diff * 0.5  # scale factor
    unet.load_state_dict(state)
    print(f"  Merged {len(pairs)} LoRA layers")


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_photo.jpg"
    style = sys.argv[2] if len(sys.argv) > 2 else "velvia_50"

    lora_path = LORAS_DIR / f"{style}_sd15.safetensors"
    if not lora_path.exists():
        print(f"No LoRA: {lora_path}"); return

    device = torch.device("mps")
    dtype = torch.float16
    caption = CAPTIONS.get(style, f"{style} film photograph")

    print(f"Merging {style} LoRA...")
    from diffusers import StableDiffusionImg2ImgPipeline

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype, use_safetensors=True, safety_checker=None,
    ).to(device)

    merge_lora(pipe.unet, str(lora_path))
    print("LoRA merged into UNet")

    img = Image.open(image_path).convert("RGB")

    t0 = time.time()
    result = pipe(
        prompt=caption,
        negative_prompt="digital, oversharpened",
        image=img, strength=0.45,
        num_inference_steps=20, guidance_scale=7.5,
    ).images[0]
    elapsed = time.time() - t0

    out = f"output_{style}.jpg"
    result.save(out, "JPEG", quality=92)
    print(f"Inference: {elapsed:.1f}s → {out}")


if __name__ == "__main__":
    main()
