#!/usr/bin/env python3
"""Execute the frozen U1.4C2 Rec.2020 real-image visual/OOD audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_engine import apply_rec2020_safe_lab, linear_rgb_to_lab  # noqa: E402
from src.eval.rec2020_visual_ood import (  # noqa: E402
    build_anonymous_sheet,
    encoded_srgb_to_linear,
    residual_gradient_quantile,
    risk_rank,
    save_srgb_preview,
    sha256_path,
    validate_filmr_selection,
)
from src.preprocess import (  # noqa: E402
    SourceProfile,
    WorkingImage,
    convert_linear_rgb,
    save_rec2020_16_png,
)


DEFAULT_CONFIG = ROOT / "configs" / "u1_4c2_rec2020_visual_ood_v1.json"


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _style_kwargs(style: str, gamut_mode: str, stats: dict, profiles: dict, guardrails: dict) -> dict:
    profile = dict(profiles["defaults"])
    profile.update(profiles["styles"].get(style, {}))
    guards = dict(guardrails["defaults"])
    guards.update(guardrails["styles"].get(style, {}))
    style_stats = stats[style]
    return {
        "destination_mean": np.asarray(style_stats["mean"], dtype=np.float32),
        "destination_std": np.asarray(style_stats["std"], dtype=np.float32),
        "style": style,
        "strength": profile["strength"],
        "luma_strength": profile["luma_strength"],
        "gamut_mode": gamut_mode,
        "tone_rolloff": profile["tone_rolloff"],
        "shadow_floor_l": profile["shadow_floor_l"],
        "highlight_ceiling_l": profile["highlight_ceiling_l"],
        "preserve_luma_detail_strength": profile["preserve_luma_detail"],
        "chroma_curve_strength": profile["chroma_curve_strength"],
        "neutral_protect": guards.get("neutral_protect", 0.0),
        "skin_protect": guards.get("skin_protect", 0.0),
        "max_chroma_gain": guards.get("max_chroma_gain"),
        "max_chroma_boost": guards.get("max_chroma_boost"),
        "max_chroma_absolute": guards.get("max_chroma_absolute"),
    }


def _working_from_unprofiled_filmr(path: Path) -> tuple[WorkingImage, np.ndarray]:
    with Image.open(path) as image:
        source = ImageOps.exif_transpose(image).convert("RGB")
        encoded = np.asarray(source, dtype=np.float32) / np.float32(255.0)
    linear_srgb = encoded_srgb_to_linear(encoded)
    rec2020 = convert_linear_rgb(
        linear_srgb,
        source_space="linear_srgb",
        destination_space="linear_rec2020",
    )
    working = WorkingImage(
        pixels=rec2020,
        working_space="linear_rec2020",
        transfer_state="display_linear",
        source_transfer_state="display_referred",
        source_profile=SourceProfile("unknown", "FILM-R unprofiled JPEG; assumed sRGB for U1.4C2 OOD audit"),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=8,
        source_path=path,
        warnings=[],
    )
    return working, linear_srgb


def _boundary_mask(pixels: np.ndarray, epsilon: float) -> np.ndarray:
    return np.any((pixels <= epsilon) | (pixels >= 1.0 - epsilon), axis=2)


def _render_record(
    *,
    working: WorkingImage,
    source_linear_srgb: np.ndarray,
    sample: dict,
    style: str,
    gamut_mode: str,
    kwargs: dict,
    masters: Path,
    previews: Path,
    epsilon: float,
    pass_id: int,
) -> dict:
    output = apply_rec2020_safe_lab(working, **kwargs)
    logical_id = f"{sample['pair_id']}__{style}__{gamut_mode}"
    if output.pixels.shape != working.pixels.shape:
        raise ValueError(f"shape changed: {logical_id}")
    if not np.isfinite(output.pixels).all():
        raise ValueError(f"non-finite output: {logical_id}")

    master_path = masters / f"{logical_id}.png"
    save_rec2020_16_png(output, master_path)
    master_sha = sha256_path(master_path)
    linear_srgb = convert_linear_rgb(
        output.pixels,
        source_space="linear_rec2020",
        destination_space="linear_srgb",
    )
    outside_srgb = np.any((linear_srgb < 0.0) | (linear_srgb > 1.0), axis=2)
    source_boundary = _boundary_mask(working.pixels, epsilon)
    output_boundary = _boundary_mask(output.pixels, epsilon)
    new_boundary = output_boundary & ~source_boundary
    source_lab = linear_rgb_to_lab(working.pixels, working_space="linear_rec2020")
    output_lab = linear_rgb_to_lab(output.pixels, working_space="linear_rec2020")
    delta_e = np.linalg.norm(output_lab - source_lab, axis=2)
    preview_path = previews / f"{logical_id}.png"
    preview_sha = None
    if pass_id == 1:
        preview_sha = save_srgb_preview(linear_srgb, preview_path)
    return {
        "logical_id": logical_id,
        "sample_id": sample["pair_id"],
        "style": style,
        "gamut_mode": gamut_mode,
        "shape": list(output.pixels.shape),
        "master_path": str(master_path.relative_to(ROOT)).replace("\\", "/"),
        "master_sha256": master_sha,
        "preview_path": str(preview_path.relative_to(ROOT)).replace("\\", "/") if pass_id == 1 else None,
        "preview_sha256": preview_sha,
        "rec2020_min": float(output.pixels.min()),
        "rec2020_max": float(output.pixels.max()),
        "new_rec2020_boundary_fraction": float(new_boundary.mean()),
        "srgb_preview_clip_fraction": float(outside_srgb.mean()),
        "srgb_max_excursion": max(float(np.max(linear_srgb - 1.0)), float(np.max(-linear_srgb))),
        "residual_gradient_q999": residual_gradient_quantile(working.pixels, output.pixels),
        "median_style_delta_e76": float(np.median(delta_e)),
        "mean_style_delta_e76": float(np.mean(delta_e)),
        "source_linear_srgb_min": float(source_linear_srgb.min()),
        "source_linear_srgb_max": float(source_linear_srgb.max()),
    }


def execute(config_path: Path, output_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_paths = validate_filmr_selection(config, root=ROOT)
    stats = json.loads((ROOT / "configs" / "film_color_stats.json").read_text(encoding="utf-8"))["styles"]
    profile_doc = OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs" / "color_rendering_profiles.yaml"), resolve=True
    )["profiles"]["safe_rich"]
    guardrails = json.loads((ROOT / "configs" / "color_guardrails.json").read_text(encoding="utf-8"))
    epsilon = float(config["automatic_gates"]["rec2020_boundary_epsilon"])

    pass_records: list[list[dict]] = []
    first_hashes: dict[str, str] = {}
    for pass_id in (1, 2):
        masters = output_root / ("masters" if pass_id == 1 else "repeat_tmp")
        previews = output_root / "previews"
        records: list[dict] = []
        for sample, source_path in zip(config["selection"]["samples"], source_paths, strict=True):
            working, source_linear_srgb = _working_from_unprofiled_filmr(source_path)
            source_preview = output_root / "sources" / f"{sample['pair_id']}.png"
            if pass_id == 1:
                save_srgb_preview(source_linear_srgb, source_preview)
            for style in config["render"]["styles"]:
                for gamut_mode in config["render"]["gamut_modes"]:
                    record = _render_record(
                        working=working,
                        source_linear_srgb=source_linear_srgb,
                        sample=sample,
                        style=style,
                        gamut_mode=gamut_mode,
                        kwargs=_style_kwargs(style, gamut_mode, stats, profile_doc, guardrails),
                        masters=masters,
                        previews=previews,
                        epsilon=epsilon,
                        pass_id=pass_id,
                    )
                    if pass_id == 1:
                        first_hashes[record["logical_id"]] = record["master_sha256"]
                    elif first_hashes[record["logical_id"]] != record["master_sha256"]:
                        raise ValueError(f"repeat master mismatch: {record['logical_id']}")
                    records.append(record)
                    if pass_id == 2:
                        (ROOT / record["master_path"]).unlink(missing_ok=True)
        pass_records.append(records)
        _atomic_json(output_root / f"pass_{pass_id}_manifest.json", records)
    (output_root / "repeat_tmp").rmdir()

    first, second = pass_records
    normalized_first = [{key: value for key, value in row.items() if key not in {"master_path", "preview_path", "preview_sha256"}} for row in first]
    normalized_second = [{key: value for key, value in row.items() if key not in {"master_path", "preview_path", "preview_sha256"}} for row in second]
    repeat_identical = normalized_first == normalized_second
    expected = int(config["automatic_gates"]["expected_renders"])
    maximum_new_boundary = max(row["new_rec2020_boundary_fraction"] for row in first)
    automatic_pass = (
        len(first) == expected
        and len(second) == expected
        and repeat_identical
        and maximum_new_boundary <= float(config["automatic_gates"]["maximum_new_rec2020_boundary_fraction"])
    )
    report = {
        "schema_version": 1,
        "node": config["node"],
        "config_sha256": sha256_path(config_path),
        "source_manifest_sha256": config["source_manifest_sha256"],
        "sources": len(source_paths),
        "renders_each_pass": len(first),
        "repeat_normalized_manifest_identical": repeat_identical,
        "maximum_new_rec2020_boundary_fraction": maximum_new_boundary,
        "maximum_srgb_preview_clip_fraction": max(row["srgb_preview_clip_fraction"] for row in first),
        "maximum_srgb_excursion": max(row["srgb_max_excursion"] for row in first),
        "median_of_style_medians_delta_e76": float(np.median([row["median_style_delta_e76"] for row in first])),
        "automatic_pass": automatic_pass,
        "claim_ceiling": config["claim_ceiling"],
        "records": first,
    }
    _atomic_json(output_root / "report.json", report)
    if not automatic_pass:
        return report

    candidates = sorted({f"{row['style']}__{row['gamut_mode']}" for row in first})
    private_mapping: dict[str, dict[str, dict[str, str]]] = {}
    for round_id in (1, 2):
        round_map: dict[str, dict[str, str]] = {}
        for sample in config["selection"]["samples"]:
            sample_id = sample["pair_id"]
            candidate_previews = {
                f"{row['style']}__{row['gamut_mode']}": ROOT / row["preview_path"]
                for row in first
                if row["sample_id"] == sample_id
            }
            if sorted(candidate_previews) != candidates:
                raise ValueError(f"candidate matrix incomplete: {sample_id}")
            round_map[sample_id] = build_anonymous_sheet(
                source_preview=output_root / "sources" / f"{sample_id}.png",
                candidate_previews=candidate_previews,
                sample_id=sample_id,
                round_id=round_id,
                output_path=output_root / "anonymous" / f"round_{round_id}" / f"{sample_id}.png",
            )
        private_mapping[f"round_{round_id}"] = round_map
    _atomic_json(output_root / "private_mapping.json", private_mapping)
    ranked = risk_rank(first, int(config["visual_protocol"]["full_resolution_risk_ranked_outputs"]))
    _atomic_json(output_root / "risk_ranked.json", ranked)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = (args.output_dir or ROOT / config["render"]["output_root"]).resolve()
    report = execute(config_path, output_root)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2, sort_keys=True))
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
