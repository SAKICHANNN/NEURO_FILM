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


def matched_controls(source, target):
    """Candidate-fitted development controls, not independent truth or inference."""
    x, y = source[::8, ::8], target[::8, ::8]
    change = float(np.abs(y - x).mean())
    params = {}
    controls = {}
    for name, grid in (
        ("saturation", np.linspace(0, 3, 121)),
        ("contrast", np.linspace(0.25, 2.5, 121)),
    ):
        scores = []
        for value in grid:
            image = simple_control(x, **{name: float(value)})
            scores.append(
                (
                    abs(float(np.abs(image - x).mean()) - change),
                    float(np.square(image - y).mean()),
                    float(value),
                )
            )
        params[name] = min(scores)[2]
        controls[name] = simple_control(source, **{name: params[name]})
    gain, offset = [], []
    for c in range(3):
        xx, yy = x[..., c].astype(float).ravel(), y[..., c].astype(float).ravel()
        variance = float(np.square(xx - xx.mean()).sum())
        a = (
            float(((xx - xx.mean()) * (yy - yy.mean())).sum() / variance)
            if variance > 1e-15
            else 0.0
        )
        gain.append(a)
        offset.append(float(yy.mean() - a * xx.mean()))
    controls["affine"] = np.clip(source * gain + offset, 0, 1).astype(np.float32)
    params.update(gain=gain, offset=offset)
    return controls, params


def detail_outputs(out, source_id, source, candidates):
    result = {}
    h, w = source.shape[:2]
    side = min(512, h, w)
    cy, cx = (h - side) // 2, (w - side) // 2
    for name, candidate in candidates.items():
        controls, params = matched_controls(source, candidate)
        record = {"parameters": params, "controls": {}}
        for label, rgb in controls.items():
            file = f"{source_id}--{name}--matched-{label}.png"
            record["controls"][label] = {
                "file": file,
                "sha256": save_png(out / file, Image.fromarray(quantize(rgb))),
                "candidate_rgb8_mae": float(
                    np.abs(quantize(candidate).astype(float) - quantize(rgb)).mean()
                ),
            }
        sheet = Image.new("RGB", (side * 2, (side + 24) * 2), "#181818")
        draw = ImageDraw.Draw(sheet)
        for idx, (label, rgb) in enumerate(
            {
                "identity": source,
                name: candidate,
                "matched contrast": controls["contrast"],
                "affine approximation": controls["affine"],
            }.items()
        ):
            px, py = idx % 2 * side, idx // 2 * (side + 24)
            sheet.paste(
                Image.fromarray(quantize(rgb[cy : cy + side, cx : cx + side])),
                (px, py + 24),
            )
            draw.text((px + 4, py + 5), label, fill="white")
        record["centre_crop_sha256"] = save_png(
            out / f"{source_id}--{name}--crop.png", sheet
        )
        result[name] = record
    return result


def compare(
    run_id: str,
    creative_config: str = "creative_looks_v2_development.json",
    detail: bool = False,
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
        "creative_looks_v2_refined_development.json",
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
        "detail": detail,
    }
    for row in selected:
        with Image.open(io.BytesIO(inputs[row["id"]])) as image:
            if image.mode != "RGB" or image.size != (row["width"], row["height"]):
                raise ValueError("decoded derivative header drift")
            image.thumbnail(
                (1600, 1600)
                if detail
                else (config["maximum_side"], config["maximum_side"]),
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
        if detail:
            report["rows"][-1]["matched_diagnostics"] = detail_outputs(
                out, row["id"], source, {name: arms[f"v2-{name}"] for name in specs}
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
    parser.add_argument("--detail", action="store_true")
    args = parser.parse_args()
    print(compare(args.run_id, args.creative_config, args.detail))
