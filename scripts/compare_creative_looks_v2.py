"""Labelled, bounded LOOK2 development comparison; never a promotion audit."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw

from scripts.pipeline_color_baseline import load_guardrail_config
from src.color_engine.creative_look_v2 import CreativeLookV2, render_creative_look_v2
from src.inference.render_contract import load_render_profile
from src.inference.style_safe_engine import render_resolved_safe_lab_rgb

CONFIG = ROOT / "configs/creative_looks_v2_photo_development_v1.json"


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def select_development(config: dict, manifest: list, parent: dict) -> list[dict]:
    included = parent["source"]["included_ids"]
    by_id = {row["id"]: row for row in manifest}
    if len(by_id) != len(manifest) or len(set(included)) != 16:
        raise ValueError("invalid parent population")
    ordered = sorted(
        (by_id[name] for name in included), key=lambda row: row["decoded_sha256"]
    )
    if [row["id"] for row in ordered[:8]] != config["development_ids"]:
        raise ValueError("development selection drift")
    if [row["id"] for row in ordered[8:]] != config["reserved_ids"]:
        raise ValueError("reserve selection drift")
    for row in ordered[:8]:
        if row["rights_scope"] != config["required_rights_scope"]:
            raise ValueError("source rights scope drift")
        if row["decoded_color_state"] != config["required_color_state"]:
            raise ValueError("source colour state drift")
    return ordered[:8]


def quantize(rgb: np.ndarray) -> np.ndarray:
    if not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
        raise ValueError("non-finite or out-of-range output")
    return np.rint(rgb.astype(np.float64) * 255).astype(np.uint8)


def simple_control(source: np.ndarray, *, saturation: float = 1, contrast: float = 1):
    luma = (
        source[..., 0] * 0.2126 + source[..., 1] * 0.7152 + source[..., 2] * 0.0722
    )[..., None]
    # Ordinary clipped sliders are intentional controls, not the candidate core.
    return np.clip((luma + saturation * (source - luma) - 0.5) * contrast + 0.5, 0, 1)


def save_png(path: Path, image: Image.Image) -> str:
    with path.open("xb") as handle:
        image.save(handle, format="PNG", compress_level=6)
    return sha(path.read_bytes())


def compare(
    run_id: str, creative_config: str = "creative_looks_v2_development.json"
) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("run id must be a simple absent directory name")
    config = read_json(CONFIG)
    manifest_bytes = (ROOT / config["manifest"]).read_bytes()
    if sha(manifest_bytes) != config["manifest_sha256"]:
        raise ValueError("manifest hash mismatch")
    parent = read_json(ROOT / config["parent_population"])
    selected = select_development(config, json.loads(manifest_bytes), parent)
    # Preflight all selected bodies BEFORE any decode; do not open reserve files.
    inputs = {}
    for row in selected:
        path = ROOT / row["decoded_path"]
        if not path.resolve().is_relative_to((ROOT / "outputs").resolve()):
            raise ValueError("source escaped outputs namespace")
        data = path.read_bytes()
        if len(data) != row["decoded_bytes"] or sha(data) != row["decoded_sha256"]:
            raise ValueError(f"source bytes drift: {row['id']}")
        inputs[row["id"]] = data

    if creative_config not in (
        "creative_looks_v2_development.json",
        "creative_looks_v2_bold_development.json",
    ):
        raise ValueError("unrecognized development config")
    creative_path = ROOT / "configs" / creative_config
    creative = read_json(creative_path)
    specs = {name: CreativeLookV2(**row) for name, row in creative["looks"].items()}
    profile = load_render_profile(
        ROOT / "configs/render_profiles/safe_rich_product_v1.json", root=ROOT
    )
    statistics = read_json(ROOT / "configs/film_color_stats.json")["styles"]
    bindings = [
        CONFIG,
        creative_path,
        Path(__file__),
        ROOT / "src/color_engine/creative_look_v2.py",
    ]
    bindings += [ROOT / row["path"] for row in profile["assets"]]
    bindings.append(ROOT / "configs/render_profiles/safe_rich_product_v1.json")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    output_parent = ROOT / config["output_parent"]
    if not output_parent.resolve().is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("output escaped project outputs")
    output_parent.mkdir(exist_ok=True)
    out = output_parent / run_id
    out.mkdir(exist_ok=False)
    report = {
        "status": "DEVELOPMENT_ONLY_NO_PROMOTION_DECISION",
        "source_commit": source_commit,
        "bindings": {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in bindings},
        "prior_exposure": config["prior_exposure"],
        "reserved_pixel_reads": 0,
        "raw_reads": 0,
        "network_reads": 0,
        "rows": [],
    }
    for row in selected:
        with Image.open(io.BytesIO(inputs[row["id"]])) as image:
            if image.mode != "RGB" or image.size != (row["width"], row["height"]):
                raise ValueError("decoded derivative header drift")
            image.thumbnail(
                (config["maximum_side"], config["maximum_side"]),
                Image.Resampling.LANCZOS,
            )
            source8 = np.asarray(image, dtype=np.uint8).copy()
        source = source8.astype(np.float32) / np.float32(255)
        arms = {"identity": source}
        for style in ("velvia_50", "portra_400", "ektar_100"):
            arms[f"v1-{style}"] = render_resolved_safe_lab_rgb(
                source,
                style=style,
                style_statistics=statistics[style],
                style_parameters=profile["style_parameters"][style],
                guardrails=load_guardrail_config(
                    ROOT / "configs/color_guardrails.json", style
                ),
                seed=42,
            )
        for name, spec in specs.items():
            arms[f"v2-{name}"] = render_creative_look_v2(
                source, spec, amount=config["amount"]
            )
        arms["basic-saturation-1.2"] = simple_control(
            source, saturation=config["simple_controls"]["saturation"]
        )
        arms["basic-contrast-1.15"] = simple_control(
            source, contrast=config["simple_controls"]["contrast"]
        )
        h, w = source.shape[:2]
        sheet = Image.new("RGB", (w * 3, (h + 24) * 3), "#181818")
        draw = ImageDraw.Draw(sheet)
        arm_rows = {}
        for index, (name, rgb) in enumerate(arms.items()):
            pixels = quantize(rgb)
            preview = Image.fromarray(pixels)
            digest = save_png(out / f"{row['id']}--{name}.png", preview)
            x, y = index % 3 * w, index // 3 * (h + 24)
            sheet.paste(preview, (x, y + 24))
            draw.text((x + 4, y + 5), name, fill="white")
            interior = (source8 > 0) & (source8 < 255)
            new_boundary = interior & ((pixels == 0) | (pixels == 255))
            arm_rows[name] = {
                "sha256": digest,
                "rgb8_mean_absolute_change": float(
                    np.abs(pixels.astype(float) - source8).mean()
                ),
                "new_endpoint_components": int(new_boundary.sum()),
                "component_count": int(pixels.size),
            }
        sheet_name = f"{row['id']}--comparison.png"
        sheet_sha = save_png(out / sheet_name, sheet)
        report["rows"].append(
            {
                "source_id": row["id"],
                "source_sha256": row["decoded_sha256"],
                "size": [w, h],
                "arms": arm_rows,
                "sheet": sheet_name,
                "sheet_sha256": sheet_sha,
            }
        )
        print(f"development {row['id']}: {len(arms)} arms", flush=True)
    with (out / "report.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=True, sort_keys=True, indent=2)
        handle.write("\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--creative-config", default="creative_looks_v2_development.json"
    )
    args = parser.parse_args()
    print(compare(args.run_id, args.creative_config))
