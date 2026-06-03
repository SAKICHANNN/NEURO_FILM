#!/usr/bin/env python3
"""Train Scheme A Style-Separated SepLUT for Neural Film LUT V2."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from skimage.color import rgb2lab
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_color_baseline import load_guardrail_config, load_profile_values, style_transfer  # noqa: E402
from src.models.film_response_volume import apply_film_response_volume  # noqa: E402
from src.models.neural_film_lut.seplut import STYLE_NAMES, StyleSeparatedSepLUT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train style-separated SepLUT.")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "outputs" / "color_baseline" / "velvia50_rawpixls20_s0p50_gamutsafe" / "manifest.json",
    )
    parser.add_argument("--stats", type=Path, default=ROOT / "configs" / "film_color_stats.json")
    parser.add_argument("--profile-config", type=Path, default=ROOT / "configs" / "color_rendering_profiles.yaml")
    parser.add_argument("--guardrails", type=Path, default=ROOT / "configs" / "color_guardrails.json")
    parser.add_argument("--teacher-root", type=Path, default=None)
    parser.add_argument("--styles", default=",".join(STYLE_NAMES))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--strengths", default="0.0,0.5,1.0,1.5")
    parser.add_argument("--lut1d-size", type=int, default=33)
    parser.add_argument("--lut3d-size", type=int, default=17)
    parser.add_argument("--reg-interval", type=int, default=4)
    parser.add_argument("--eval-loss-limit", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "neural_film_lut_v2" / "scheme_a_seplut")
    return parser.parse_args()


def parse_styles(raw: str) -> list[str]:
    styles = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [style for style in styles if style not in STYLE_NAMES]
    if unknown:
        raise ValueError(f"Unknown styles: {unknown}")
    return styles


def parse_strengths(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if 0.0 not in values:
        values = [0.0, *values]
    return sorted(set(values))


def load_source_paths(path: Path, limit: int) -> list[Path]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    paths = []
    for row in rows[:limit]:
        before = Path(row.get("before") or row.get("input") or "")
        if before.exists():
            paths.append(before)
    if not paths:
        raise ValueError(f"No source images found in {path}")
    return paths


def load_image(path: Path, size: int) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "black")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def rgb_np_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.clip(rgb, 0.0, 1.0).astype(np.float32)).permute(2, 0, 1)


def skin_neutral_masks(image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    lab = rgb2lab(np.asarray(image, dtype=np.float32) / 255.0)
    chroma = np.linalg.norm(lab[..., 1:3], axis=2)
    neutral = (chroma < 8.0).astype(np.float32)
    skin = (
        (lab[..., 0] > 20.0)
        & (lab[..., 0] < 92.0)
        & (lab[..., 1] > 4.0)
        & (lab[..., 1] < 28.0)
        & (lab[..., 2] > 4.0)
        & (lab[..., 2] < 46.0)
    ).astype(np.float32)
    return torch.from_numpy(neutral)[None], torch.from_numpy(skin)[None]


def safe_rich_image(image: Image.Image, stats: dict, style: str, args: argparse.Namespace, index: int) -> Image.Image:
    profile = load_profile_values(args.profile_config, "safe-rich", style)
    return style_transfer(
        image,
        stats["styles"][style],
        style,
        strength=profile["strength"],
        luma_strength=profile["luma_strength"],
        grain=profile["grain"],
        seed=7 + index,
        gamut_safe=profile["gamut_safe"],
        gamut_mode=profile["gamut_mode"],
        tone_rolloff=profile["tone_rolloff"],
        shadow_floor_l=profile["shadow_floor_l"],
        highlight_ceiling_l=profile["highlight_ceiling_l"],
        preserve_luma_detail_strength=profile["preserve_luma_detail"],
        chroma_curve_strength=profile["chroma_curve_strength"],
        output_margin=profile["output_margin"],
        guardrails=load_guardrail_config(args.guardrails, style),
        dither=0.0,
    )


def make_target_from_teacher(source: torch.Tensor, safe: torch.Tensor, response: torch.Tensor, strength: float) -> torch.Tensor:
    if strength <= 0.0:
        return source
    # Blend through safe-rich so moderate strengths remain well-behaved.
    mix = min(1.0, strength / 1.25)
    return safe * (1.0 - mix) + response * mix


def build_examples(args: argparse.Namespace, styles: list[str], strengths: list[float]) -> list[dict]:
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    paths = load_source_paths(args.source_manifest, args.limit)
    examples = []
    for image_index, path in enumerate(paths, start=1):
        source_image = load_image(path, args.image_size)
        source = image_to_tensor(source_image)
        neutral, skin = skin_neutral_masks(source_image)
        for style in styles:
            safe_path = (
                args.teacher_root / style / "safe_rich" / f"{image_index:02d}_{style}_safe_rich.png"
                if args.teacher_root
                else None
            )
            response_path = (
                args.teacher_root / style / "after" / f"{image_index:02d}_{style}_film_response.png"
                if args.teacher_root
                else None
            )
            safe_image = (
                load_image(safe_path, args.image_size)
                if safe_path and safe_path.exists()
                else safe_rich_image(source_image, stats, style, args, image_index)
            )
            if response_path and response_path.exists():
                response = image_to_tensor(load_image(response_path, args.image_size))
            else:
                source_rgb = np.asarray(source_image, dtype=np.float32) / 255.0
                safe_rgb = np.asarray(safe_image, dtype=np.float32) / 255.0
                response_rgb, _metrics = apply_film_response_volume(
                    source_rgb,
                    safe_rgb,
                    style=style,
                    strength=1.25,
                    output_margin=4,
                )
                response = rgb_np_to_tensor(response_rgb)
            safe = image_to_tensor(safe_image)
            for strength in strengths:
                examples.append(
                    {
                        "path": str(path),
                        "style": style,
                        "style_index": STYLE_NAMES.index(style),
                        "strength": float(strength),
                        "source": source,
                        "target": make_target_from_teacher(source, safe, response, float(strength)),
                        "neutral": neutral,
                        "skin": skin,
                    }
                )
    return examples


def batch_items(items: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "source": torch.stack([item["source"] for item in items]).to(device),
        "target": torch.stack([item["target"] for item in items]).to(device),
        "style_index": torch.tensor([item["style_index"] for item in items], device=device),
        "strength": torch.tensor([item["strength"] for item in items], device=device, dtype=torch.float32),
        "neutral": torch.stack([item["neutral"] for item in items]).to(device),
        "skin": torch.stack([item["skin"] for item in items]).to(device),
    }


def style_separation_loss(model: StyleSeparatedSepLUT, sample: dict, styles: list[str], device: torch.device) -> torch.Tensor:
    source = sample["source"].unsqueeze(0).to(device).repeat(len(styles), 1, 1, 1)
    style_index = torch.tensor([STYLE_NAMES.index(style) for style in styles], device=device)
    strength = torch.full((len(styles),), 1.0, device=device)
    outputs = model(source, style_index, strength)
    flattened = outputs.mean(dim=(2, 3))
    distances = []
    for i in range(len(styles)):
        for j in range(i + 1, len(styles)):
            distances.append(torch.abs(flattened[i] - flattened[j]).mean())
    if not distances:
        return torch.tensor(0.0, device=device)
    mean_distance = torch.stack(distances).mean()
    return torch.relu(0.018 - mean_distance)


def eval_loss(model: StyleSeparatedSepLUT, examples: list[dict], device: torch.device, loss_fn: nn.Module) -> float:
    with torch.no_grad():
        losses = []
        for item in examples:
            batch = batch_items([item], device)
            pred = model(batch["source"], batch["style_index"], batch["strength"])
            losses.append(loss_fn(pred, batch["target"]).item())
        return float(sum(losses) / len(losses))


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    styles = parse_styles(args.styles)
    strengths = parse_strengths(args.strengths)
    examples = build_examples(args, styles, strengths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StyleSeparatedSepLUT(
        style_count=len(STYLE_NAMES),
        lut1d_size=args.lut1d_size,
        lut3d_size=args.lut3d_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()
    print(f"built_examples={len(examples)} device={device}")
    eval_subset = examples[: max(1, min(args.eval_loss_limit, len(examples)))]
    initial_loss = eval_loss(model, eval_subset, device, loss_fn)

    for step in range(1, args.steps + 1):
        items = random.sample(examples, k=min(args.batch_size, len(examples)))
        batch = batch_items(items, device)
        pred = model(batch["source"], batch["style_index"], batch["strength"])
        target_loss = loss_fn(pred, batch["target"])
        neutral_loss = (torch.abs(pred - batch["source"]) * batch["neutral"]).mean()
        skin_loss = (torch.abs(pred - batch["source"]) * batch["skin"]).mean()
        margin = 4.0 / 255.0
        clip_loss = torch.relu(margin - pred).mean() + torch.relu(pred - (1.0 - margin)).mean()
        if step % max(1, args.reg_interval) == 0:
            regs = model.regularization()
            sep_loss = style_separation_loss(model, random.choice(examples), styles, device)
        else:
            zero = torch.tensor(0.0, device=device)
            regs = {"monotonic_1d": zero, "smooth_1d": zero, "smooth_3d": zero}
            sep_loss = zero
        loss = (
            target_loss
            + 0.20 * neutral_loss
            + 0.12 * skin_loss
            + 2.0 * clip_loss
            + 0.50 * regs["monotonic_1d"]
            + 0.10 * regs["smooth_1d"]
            + 0.05 * regs["smooth_3d"]
            + 0.25 * sep_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(
                " ".join(
                    [
                        f"step={step}",
                        f"loss={loss.item():.6f}",
                        f"target={target_loss.item():.6f}",
                        f"sep={sep_loss.item():.6f}",
                        f"mono={regs['monotonic_1d'].item():.6f}",
                    ]
                )
            )

    final_loss = eval_loss(model, eval_subset, device, loss_fn)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "styles": STYLE_NAMES,
        "train_styles": styles,
        "strengths": strengths,
        "lut1d_size": args.lut1d_size,
        "lut3d_size": args.lut3d_size,
        "args": vars(args),
    }
    torch.save(checkpoint, output_dir / "model.pt")
    metrics = {
        "device": str(device),
        "styles": styles,
        "strengths": strengths,
        "example_count": len(examples),
        "steps": args.steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "improvement_ratio": initial_loss / max(final_loss, 1e-8),
        "output_dir": str(output_dir),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
