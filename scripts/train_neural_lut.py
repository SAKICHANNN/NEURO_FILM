#!/usr/bin/env python3
"""Train a Neural LUT MVP to imitate the safe-rich deterministic renderer."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_color_baseline import load_guardrail_config, load_profile_values, style_transfer  # noqa: E402
from src.models.color_lut import BasisLUT, TinyLUTEncoder, apply_lut  # noqa: E402


STYLE_NAMES = [
    "ektar_100",
    "hp5",
    "portra_400",
    "portra_800",
    "tri_x_400",
    "velvia_50",
    "vision3_250d",
    "vision3_500t",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Neural LUT imitation MVP.")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "outputs" / "color_baseline" / "velvia50_rawpixls20_s0p50_gamutsafe" / "manifest.json",
    )
    parser.add_argument("--stats", type=Path, default=ROOT / "configs" / "film_color_stats.json")
    parser.add_argument("--profile-config", type=Path, default=ROOT / "configs" / "color_rendering_profiles.yaml")
    parser.add_argument("--guardrails", type=Path, default=ROOT / "configs" / "color_guardrails.json")
    parser.add_argument("--styles", default="portra_400,velvia_50")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lut-size", type=int, default=17)
    parser.add_argument("--num-basis", type=int, default=4)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "neural_lut" / "mvp_smoke")
    return parser.parse_args()


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


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.rint(arr * 255.0).astype(np.uint8), mode="RGB")


def parse_styles(raw: str) -> list[str]:
    styles = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [style for style in styles if style not in STYLE_NAMES]
    if unknown:
        raise ValueError(f"Unknown styles: {unknown}")
    return styles


def make_target(image: Image.Image, stats: dict, style: str, profile: dict, guardrail_path: Path) -> Image.Image:
    return style_transfer(
        image,
        stats["styles"][style],
        style,
        strength=profile["strength"],
        luma_strength=profile["luma_strength"],
        grain=profile["grain"],
        seed=11,
        gamut_safe=profile["gamut_safe"],
        gamut_mode=profile["gamut_mode"],
        tone_rolloff=profile["tone_rolloff"],
        shadow_floor_l=profile["shadow_floor_l"],
        highlight_ceiling_l=profile["highlight_ceiling_l"],
        preserve_luma_detail_strength=profile["preserve_luma_detail"],
        chroma_curve_strength=profile["chroma_curve_strength"],
        output_margin=profile["output_margin"],
        guardrails=load_guardrail_config(guardrail_path, style),
        dither=0.0,
    )


def build_examples(args: argparse.Namespace, styles: list[str]) -> list[dict]:
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    paths = load_source_paths(args.source_manifest, args.limit)
    examples = []
    for path in paths:
        image = load_image(path, args.image_size)
        source = image_to_tensor(image)
        for style in styles:
            profile = load_profile_values(args.profile_config, "safe-rich", style)
            target = image_to_tensor(make_target(image, stats, style, profile, args.guardrails))
            examples.append(
                {
                    "path": str(path),
                    "style": style,
                    "style_index": STYLE_NAMES.index(style),
                    "source": source,
                    "target": target,
                }
            )
    return examples


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    styles = parse_styles(args.styles)
    examples = build_examples(args, styles)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    basis = BasisLUT(size=args.lut_size, num_basis=args.num_basis).to(device)
    encoder = TinyLUTEncoder(num_basis=args.num_basis, style_count=len(STYLE_NAMES)).to(device)
    optimizer = torch.optim.AdamW(list(basis.parameters()) + list(encoder.parameters()), lr=args.lr)
    loss_fn = nn.L1Loss()

    def eval_loss() -> float:
        with torch.no_grad():
            losses = []
            for item in examples:
                source = item["source"].unsqueeze(0).to(device)
                target = item["target"].unsqueeze(0).to(device)
                style_index = torch.tensor([item["style_index"]], device=device)
                pred = apply_lut(source, basis(encoder(source, style_index)))
                losses.append(loss_fn(pred, target).item())
            return float(sum(losses) / len(losses))

    initial_loss = eval_loss()
    for step in range(1, args.steps + 1):
        item = random.choice(examples)
        source = item["source"].unsqueeze(0).to(device)
        target = item["target"].unsqueeze(0).to(device)
        style_index = torch.tensor([item["style_index"]], device=device)
        pred = apply_lut(source, basis(encoder(source, style_index)))
        loss = loss_fn(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 20 == 0:
            print(f"step={step} loss={loss.item():.6f}")

    final_loss = eval_loss()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = examples[0]
    with torch.no_grad():
        source = sample["source"].unsqueeze(0).to(device)
        style_index = torch.tensor([sample["style_index"]], device=device)
        pred = apply_lut(source, basis(encoder(source, style_index))).squeeze(0)
    tensor_to_image(sample["source"]).save(output_dir / "sample_source.png", "PNG")
    tensor_to_image(sample["target"]).save(output_dir / "sample_target_safe_rich.png", "PNG")
    tensor_to_image(pred).save(output_dir / "sample_pred_neural_lut.png", "PNG")
    metrics = {
        "device": str(device),
        "styles": styles,
        "example_count": len(examples),
        "steps": args.steps,
        "initial_l1": initial_loss,
        "final_l1": final_loss,
        "improvement_ratio": initial_loss / max(final_loss, 1e-8),
        "sample_style": sample["style"],
        "output_dir": str(output_dir),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"basis": basis.state_dict(), "encoder": encoder.state_dict(), "styles": STYLE_NAMES}, output_dir / "model.pt")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
