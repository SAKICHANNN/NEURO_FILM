"""Labelled, bounded LOOK2 development comparison; never a promotion audit."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw

from scripts.pipeline_color_baseline import load_guardrail_config
from src.color_engine.creative_hue_look import CreativeHueLook, render_creative_hue_look
from src.color_engine.creative_look_v2 import CreativeLookV2, render_creative_look_v2
from src.color_engine.creative_ordered_hue_look import (
    OrderedHueLook,
    render_ordered_hue_look,
)
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


def select_ordered_assessment(lock, manifest, previous, review):
    """Select an intact historical pool without reading any image body."""
    by_id = {row["id"]: row for row in manifest}
    ids = lock["included_ids"]
    if len(by_id) != len(manifest) or len(ids) != 10 or len(set(ids)) != 10:
        raise ValueError("invalid independent population")
    if ids != review["eligible_ids"]:
        raise ValueError("historical eligibility drift")
    selected = [by_id[name] for name in ids]
    seen = []
    for row in selected:
        if (
            row["rights_scope"] != "CC0_public_domain_internal_evaluation"
            or row["decoded_color_state"] != "relative_display_srgb_approximation"
        ):
            raise ValueError("assessment rights/colour drift")
        for other in previous + seen:
            if (
                row["id"] == other["id"]
                or row["decoded_sha256"] == other["decoded_sha256"]
                or (int(row["dhash64"], 16) ^ int(other["dhash64"], 16)).bit_count()
                <= 4
            ):
                raise ValueError("assessment source overlap")
        seen.append(row)
    return selected


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
    assessment: bool = False,
    subject_detail: bool = False,
    ordered_assessment: bool = False,
) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("run id must be a simple absent directory name")
    config = read_json(CONFIG)
    if ordered_assessment and (assessment or detail or subject_detail):
        raise ValueError("ordered assessment is a separate fixed comparison")
    manifest_bytes = (ROOT / config["manifest"]).read_bytes()
    if sha(manifest_bytes) != config["manifest_sha256"]:
        raise ValueError("manifest hash mismatch")
    parent = read_json(ROOT / config["parent_population"])
    selected = select_development(config, json.loads(manifest_bytes), parent)
    if creative_config not in (
        "creative_looks_v2_development.json",
        "creative_looks_v2_bold_development.json",
        "creative_looks_v2_refined_development.json",
        "creative_hue_look_development_v1.json",
        "creative_hue_look_development_v2.json",
        "creative_ordered_hue_development_v1.json",
        "creative_ordered_hue_development_v2.json",
    ):
        raise ValueError("unrecognized development config")
    ordered_mode = creative_config.startswith("creative_ordered_hue_development_")
    hue_mode = (
        creative_config.startswith("creative_hue_look_development_") or ordered_mode
    )
    if subject_detail and (assessment or detail or not hue_mode or ordered_mode):
        raise ValueError("subject detail is a separate hue-development diagnostic")
    creative_path = ROOT / "configs" / creative_config
    creative = read_json(creative_path)
    if hue_mode and not assessment and not ordered_assessment:
        if (
            creative["additional_development"]["role"]
            != "previously-consumed-development-only"
        ):
            raise ValueError("additional source must remain development-only")
        selected.append(creative["additional_development"])
    lock = None
    if ordered_assessment:
        lock = read_json(ROOT / "configs/creative_ordered_hue_assessment_v1.json")
        if creative_config != lock["candidate_config"]:
            raise ValueError("ordered assessment candidate mismatch")
        for path, expected in (
            (creative_path, lock["candidate_sha256"]),
            (
                ROOT / "src/color_engine/creative_ordered_hue_look.py",
                lock["core_sha256"],
            ),
            (ROOT / lock["manifest"], lock["manifest_sha256"]),
            (ROOT / lock["source_review"], lock["source_review_sha256"]),
        ):
            if sha(path.read_bytes()) != expected:
                raise ValueError("ordered assessment binding drift before pixels")
        selected = select_ordered_assessment(
            lock,
            read_json(ROOT / lock["manifest"]),
            json.loads(manifest_bytes),
            read_json(ROOT / lock["source_review"]),
        )
    if assessment:
        lock = read_json(ROOT / "configs/creative_looks_v2_assessment_v1.json")
        if creative_config != lock["candidate_config"]:
            raise ValueError("assessment candidate mismatch")
        for path, expected in (
            (ROOT / "configs" / creative_config, lock["candidate_sha256"]),
            (ROOT / "src/color_engine/creative_look_v2.py", lock["core_sha256"]),
        ):
            if sha(path.read_bytes()) != expected:
                raise ValueError("frozen candidate drift before image access")
        by_id = {r["id"]: r for r in json.loads(manifest_bytes)}
        selected = [by_id[name] for name in config["reserved_ids"]]
        if any(
            r["rights_scope"] != config["required_rights_scope"]
            or r["decoded_color_state"] != config["required_color_state"]
            for r in selected
        ):
            raise ValueError("assessment source rights/colour drift")
    if subject_detail:
        diagnostic_ids = {"sony_dslr_a290", "leica_d_lux_6", "canon_eos_1d_mark_iv"}
        selected = [r for r in selected if r["id"] in diagnostic_ids]
        if {r["id"] for r in selected} != diagnostic_ids:
            raise ValueError("subject diagnostic source selection drift")
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

    spec_class = CreativeHueLook if hue_mode else CreativeLookV2
    render = render_creative_hue_look if hue_mode else render_creative_look_v2
    if ordered_mode:
        spec_class, render = OrderedHueLook, render_ordered_hue_look
    arm_prefix = "hue" if hue_mode else "v2"
    specs = {name: spec_class(**row) for name, row in creative["looks"].items()}
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
    if hue_mode:
        bindings.append(ROOT / "src/color_engine/creative_hue_look.py")
    if ordered_mode:
        bindings.append(ROOT / "src/color_engine/creative_ordered_hue_look.py")
    bindings += [ROOT / row["path"] for row in profile["assets"]]
    if assessment:
        bindings.append(ROOT / "configs/creative_looks_v2_assessment_v1.json")
    if ordered_assessment:
        bindings.append(ROOT / "configs/creative_ordered_hue_assessment_v1.json")
        bindings.extend([ROOT / lock["manifest"], ROOT / lock["source_review"]])
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
        "status": "INDEPENDENT_LANDSCAPE_ASSESSMENT_PENDING_VISUAL_REVIEW"
        if ordered_assessment
        else "RESERVED_ASSESSMENT_PENDING_VISUAL_REVIEW"
        if assessment
        else "DEVELOPMENT_ONLY_NO_PROMOTION_DECISION",
        "source_commit": source_commit,
        "bindings": {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in bindings},
        "prior_exposure": lock["prior_exposure"]
        if ordered_assessment
        else config["prior_exposure"],
        "reserved_pixel_reads": len(selected) if assessment else 0,
        "raw_reads": 0,
        "network_reads": 0,
        "rows": [],
        "detail": detail,
    }
    if hue_mode and not ordered_assessment:
        report["additional_development"] = creative["additional_development"]
    if subject_detail:
        report["subject_detail"] = "three-known-development-rows-tone-colour-ablation"
    for row in selected:
        with Image.open(io.BytesIO(inputs[row["id"]])) as image:
            if image.mode != "RGB" or image.size != (row["width"], row["height"]):
                raise ValueError("decoded derivative header drift")
            image.thumbnail(
                (lock["maximum_side"], lock["maximum_side"])
                if assessment or ordered_assessment
                else (1600, 1600)
                if detail or subject_detail
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
            arms[f"{arm_prefix}-{name}"] = render(source, spec, amount=config["amount"])
            if ordered_mode and spec.value_lift:
                arms[f"lift-only-{name}"] = render(
                    source,
                    replace(
                        spec, mapped_hue=spec.source_hue, green_logsat=0, blue_logsat=0
                    ),
                    amount=config["amount"],
                )
                arms[f"colour-only-{name}"] = render(
                    source, replace(spec, value_lift=0), amount=config["amount"]
                )
            if subject_detail:
                arms[f"tone-only-{name}"] = render(
                    source,
                    replace(
                        spec, green_shift=0, blue_shift=0, green_logsat=0, blue_logsat=0
                    ),
                    amount=config["amount"],
                )
                arms[f"colour-only-{name}"] = render(
                    source,
                    replace(spec, tone=(1 / 3, 2 / 3)),
                    amount=config["amount"],
                )
        arms["basic-saturation-1.2"] = simple_control(
            source, saturation=config["simple_controls"]["saturation"]
        )
        arms["basic-contrast-1.15"] = simple_control(
            source, contrast=config["simple_controls"]["contrast"]
        )
        h, w = source.shape[:2]
        if assessment:
            for name in specs:
                target = arms[f"v2-{name}"]
                controls, _ = matched_controls(source, target)
                arms[f"affine-{name}"] = controls["affine"]
                x, y = source[::8, ::8], target[::8, ::8]
                scores = [
                    (
                        float(
                            np.square(
                                simple_control(x, saturation=s, contrast=c) - y
                            ).mean()
                        ),
                        float(s),
                        float(c),
                    )
                    for s in np.linspace(0, 2, 21)
                    for c in np.linspace(0.5, 1.5, 21)
                ]
                _, s, c = min(scores)
                arms[f"closest-basic-{name}"] = simple_control(
                    source, saturation=s, contrast=c
                )
        sw, sh = w, h
        if ordered_assessment:
            ratio = min(1, lock["sheet_side"] / max(w, h))
            sw, sh = round(w * ratio), round(h * ratio)
        sheet = Image.new(
            "RGB", (sw * 3, (sh + 24) * ((len(arms) + 2) // 3)), "#181818"
        )
        draw = ImageDraw.Draw(sheet)
        arm_rows = {}
        for index, (name, rgb) in enumerate(arms.items()):
            pixels = quantize(rgb)
            preview = Image.fromarray(pixels)
            digest = save_png(out / f"{row['id']}--{name}.png", preview)
            x, y = index % 3 * sw, index // 3 * (sh + 24)
            sheet.paste(preview.resize((sw, sh), Image.Resampling.LANCZOS), (x, y + 24))
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
                out,
                row["id"],
                source,
                {name: arms[f"{arm_prefix}-{name}"] for name in specs},
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
    parser.add_argument("--assessment", action="store_true")
    parser.add_argument("--subject-detail", action="store_true")
    parser.add_argument("--ordered-assessment", action="store_true")
    args = parser.parse_args()
    print(
        compare(
            args.run_id,
            args.creative_config,
            args.detail,
            args.assessment,
            args.subject_detail,
            args.ordered_assessment,
        )
    )
