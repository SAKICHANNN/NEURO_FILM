#!/usr/bin/env python3
"""Evaluate a trained Neural LUT checkpoint with Part 1 safety metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_render_safety import evaluate, load_rgb_u8, make_diff_map  # noqa: E402
from src.models.color_lut import BasisLUT, TinyLUTEncoder, apply_lut  # noqa: E402


COLOR_STYLES = ["ektar_100", "portra_400", "portra_800", "velvia_50", "vision3_250d", "vision3_500t"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Neural LUT outputs.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "outputs" / "color_baseline" / "velvia50_rawpixls20_s0p50_gamutsafe" / "manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "eval" / "color_engine_challenge" / "neural_lut")
    parser.add_argument("--styles", default=",".join(COLOR_STYLES))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-margin", type=int, default=4)
    return parser.parse_args()


def source_paths(path: Path, limit: int) -> list[Path]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if limit > 0:
        rows = rows[:limit]
    return [Path(row.get("before") or row.get("input")) for row in rows if Path(row.get("before") or row.get("input")).exists()]


def load_image(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def tensor_to_rgb(tensor: torch.Tensor, output_margin: int) -> np.ndarray:
    out = tensor.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    if output_margin > 0:
        low = output_margin / 255.0
        high = 1.0 - low
        out = np.clip(out, low, high)
    return out.astype(np.float32)


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(path, "PNG")


def summarize(rows: list[dict]) -> dict:
    return {
        "image_count": len(rows),
        "new_clip_image_count": sum(1 for row in rows if row["new_clipped_pixel_count"] > 0),
        "hard_bound_image_count": sum(1 for row in rows if row["after_min"] <= 0 or row["after_max"] >= 255),
        "new_clipped_pixel_count_total": int(sum(row["new_clipped_pixel_count"] for row in rows)),
        "l_ssim_min": min(row["l_ssim"] for row in rows),
        "l_ssim_mean": mean(row["l_ssim"] for row in rows),
        "gradient_delta_p95_mean": mean(row["gradient_delta_p95"] for row in rows),
        "high_frequency_delta_p95_mean": mean(row["high_frequency_delta_p95"] for row in rows),
        "after_chroma_mean": mean(row["after_chroma_mean"] for row in rows),
        "neutral_contaminated_percent_max": max(row["neutral_contaminated_percent"] for row in rows),
        "after_min_min": min(row["after_min"] for row in rows),
        "after_max_max": max(row["after_max"] for row in rows),
    }


def contact_sheet(rows: list[dict], path: Path, title: str) -> None:
    font = ImageFont.load_default()
    thumbs = []
    for row in rows:
        before = Image.open(row["before"]).convert("RGB")
        after = Image.open(row["after"]).convert("RGB")
        for image in (before, after):
            image.thumbnail((220, 170), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (440, 204), "white")
        draw = ImageDraw.Draw(tile)
        draw.text((8, 8), f"{row['id']} L={row['l_ssim']:.5f} C={row['after_chroma_mean']:.2f}", fill="black", font=font)
        tile.paste(before, ((220 - before.width) // 2, 34))
        tile.paste(after, (220 + (220 - after.width) // 2, 34))
        thumbs.append(tile)
    sheet = Image.new("RGB", (440, len(thumbs) * 204 + 30), "white")
    ImageDraw.Draw(sheet).text((8, 8), title, fill="black", font=font)
    for index, tile in enumerate(thumbs):
        sheet.paste(tile, (0, 30 + index * 204))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, "PNG")


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[BasisLUT, TinyLUTEncoder, list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    basis_state = checkpoint["basis"]
    encoder_state = checkpoint["encoder"]
    basis_tensor = basis_state["basis"]
    num_basis = basis_tensor.shape[0]
    lut_size = basis_tensor.shape[1]
    style_count = encoder_state["style_embedding.weight"].shape[0]
    styles = checkpoint["styles"]
    basis = BasisLUT(size=lut_size, num_basis=num_basis).to(device)
    encoder = TinyLUTEncoder(num_basis=num_basis, style_count=style_count).to(device)
    basis.load_state_dict(basis_state)
    encoder.load_state_dict(encoder_state)
    basis.eval()
    encoder.eval()
    return basis, encoder, styles


def main() -> int:
    args = parse_args()
    styles = [item.strip() for item in args.styles.split(",") if item.strip()]
    paths = source_paths(args.source_manifest, args.limit)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    basis, encoder, style_names = load_model(args.checkpoint, device)
    output_dir = args.output_dir.resolve()
    summary = {
        "engine": "neural_lut",
        "checkpoint": str(args.checkpoint),
        "source_manifest": str(args.source_manifest),
        "image_count": len(paths),
        "output_margin": args.output_margin,
        "device": str(device),
        "styles": {},
    }

    with torch.no_grad():
        for style in styles:
            rows: list[dict] = []
            style_dir = output_dir / style
            style_index = torch.tensor([style_names.index(style)], device=device)
            for index, input_path in enumerate(paths, start=1):
                image = load_image(input_path)
                source = image_to_tensor(image).unsqueeze(0).to(device)
                lut = basis(encoder(source, style_index))
                pred = apply_lut(source, lut).squeeze(0)
                rgb = tensor_to_rgb(pred, args.output_margin)
                after_path = style_dir / "after" / f"{index:02d}_{style}_neural_lut.png"
                diff_path = style_dir / "diff_maps" / f"{index:02d}_{style}_diff.png"
                save_rgb(rgb, after_path)
                metrics = evaluate(input_path, after_path)
                make_diff_map(load_rgb_u8(input_path), load_rgb_u8(after_path), diff_path)
                row = {
                    "id": f"{index:02d}",
                    "before": str(input_path),
                    "after": str(after_path),
                    "diff_map": str(diff_path),
                    "after_min": metrics["clip"]["after_min"],
                    "after_max": metrics["clip"]["after_max"],
                    "new_clipped_pixel_count": metrics["clip"]["new_clipped_pixel_count"],
                    "new_clipped_pixel_percent": metrics["clip"]["new_clipped_pixel_percent"],
                    "l_ssim": metrics["structure"]["l_ssim"],
                    "gradient_delta_p95": metrics["structure"]["gradient_delta_p95"],
                    "high_frequency_delta_p95": metrics["structure"]["high_frequency_delta_p95"],
                    "after_chroma_mean": metrics["color"]["after_chroma_mean"],
                    "neutral_contaminated_percent": metrics["color"]["neutral_contaminated_percent"],
                }
                rows.append(row)
                print(f"{style} {index:02d} Lssim={row['l_ssim']:.5f} chroma={row['after_chroma_mean']:.2f}")
                del source, lut, pred
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            style_dir.mkdir(parents=True, exist_ok=True)
            with (style_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (style_dir / "metrics.json").write_text(
                json.dumps({"summary": summarize(rows), "images": rows}, indent=2),
                encoding="utf-8",
            )
            contact_sheet(rows, style_dir / "contact_sheet.png", f"{style} neural lut")
            summary["styles"][style] = summarize(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
