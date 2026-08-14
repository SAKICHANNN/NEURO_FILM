"""Strict opt-in official-ROMM to Rec.2020 Velvia look renderer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.color_engine.rec2020_safe_lab import apply_rec2020_safe_lab
from src.preprocess import (
    FIVEK_PROPHOTO_MATRIX_SHAPER_ICC_SHA256,
    OFFICIAL_ROMM_ICC_SHA256,
    REC2020_SDR_CICP,
    linear_rec2020_to_rec2020,
    load_and_map_official_romm_rgb16,
    load_and_map_supported_prophoto_rgb16,
    save_rec2020_16_png,
)
from src.preprocess.types import DecodeWarning, WorkingImage

PROFILE_SCHEMA = "kmcfm.romm-rec2020-render-profile.v1"
PROFILE_ID = "romm-rec2020-velvia-look-v1"
RECEIPT_SCHEMA = "kmcfm.romm-rec2020-render-receipt.v1"
PROPHOTO_PROFILE_SCHEMA = "kmcfm.prophoto-rec2020-render-profile.v1"
PROPHOTO_PROFILE_ID = "prophoto-rec2020-velvia-look-v1"
PROPHOTO_RECEIPT_SCHEMA = "kmcfm.prophoto-rec2020-render-receipt.v1"

_PROFILE_SPECS = {
    PROFILE_ID: {
        "schema": PROFILE_SCHEMA,
        "input_profile_binding": {"embedded_icc_sha256": OFFICIAL_ROMM_ICC_SHA256},
        "qualification_ids": {"U1.4C13", "U1.4C14"},
    },
    PROPHOTO_PROFILE_ID: {
        "schema": PROPHOTO_PROFILE_SCHEMA,
        "input_profile_binding": {
            "embedded_icc_sha256s": [FIVEK_PROPHOTO_MATRIX_SHAPER_ICC_SHA256]
        },
        "qualification_ids": {"U1.4C9", "U1.4C11", "U1.4C15"},
    },
}


class ROMMRec2020RenderError(RuntimeError):
    """Raised when the strict wide-gamut look boundary is not satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_profile(path: Path, *, root: Path) -> tuple[dict[str, Any], str]:
    path = Path(path)
    profile_sha256 = _sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    profile_id = payload.get("profile_id")
    spec = _PROFILE_SPECS.get(profile_id)
    if spec is None or payload.get("schema_id") != spec["schema"]:
        raise ROMMRec2020RenderError("unsupported ProPhoto Rec.2020 render profile")
    expected = {
        "input": {
            **spec["input_profile_binding"],
            "format": "RGB16 TIFF",
            "working_space_after_decode": "linear_rec2020",
            "transfer_state": "display_linear",
        },
        "ingress_mapper": {
            "id": "oklab-oog-only-soft-interval-maximum-chroma-v1",
            "softness": 1.0 / 64.0,
            "rgb16_margin": 2.0 / 65535.0,
            "bisection_iterations": 24,
        },
        "style": {
            "id": "velvia_50",
            "strength": 0.35,
            "luma_strength": 0.02,
            "gamut_mode": "source",
            "tone_rolloff": 0.04,
            "shadow_floor_l": 1.0,
            "highlight_ceiling_l": 99.0,
            "preserve_luma_detail_strength": 0.9,
            "chroma_curve_strength": 0.45,
        },
        "residual_execution": {
            "id": "source-anchored-rgb16-interior-residual-v1",
            "rgb16_margin": 2.0 / 65535.0,
        },
        "effects": {"grain": False, "halation": False, "dust": False},
        "output": {
            "format": "PNG",
            "bit_depth": 16,
            "working_space": "linear_rec2020",
            "transfer": "BT.2020 SDR",
            "cicp": REC2020_SDR_CICP.hex(),
        },
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ROMMRec2020RenderError(f"ROMM Rec.2020 profile {key} drift")
    qualification = payload.get("qualification", [])
    assets = payload.get("assets", [])
    if {item.get("experiment_id") for item in qualification} != spec[
        "qualification_ids"
    ]:
        raise ROMMRec2020RenderError("ProPhoto Rec.2020 qualification inventory drift")
    if {item.get("role") for item in assets} != {"style_statistics", "color_guardrails"}:
        raise ROMMRec2020RenderError("ROMM Rec.2020 asset inventory drift")
    for binding in [*qualification, *assets]:
        relative = Path(binding["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ROMMRec2020RenderError("ROMM Rec.2020 profile path is not relative")
        asset = root / relative
        if not asset.is_file() or _sha256(asset) != binding["sha256"]:
            raise ROMMRec2020RenderError("ROMM Rec.2020 profile binding drift")
        if "required_status" in binding:
            evidence = json.loads(asset.read_text(encoding="utf-8"))
            if evidence.get("status") != binding["required_status"]:
                raise ROMMRec2020RenderError("ROMM Rec.2020 qualification status drift")
    return payload, profile_sha256


def _source_anchored_interior_residual(
    source: np.ndarray, candidate: np.ndarray, *, margin: float
) -> tuple[np.ndarray, np.ndarray]:
    if (
        source.dtype != np.float32
        or candidate.dtype != np.float32
        or source.ndim != 3
        or source.shape[-1] != 3
        or candidate.shape != source.shape
        or not np.isfinite(source).all()
        or not np.isfinite(candidate).all()
        or float(np.min(source)) < 0.0
        or float(np.max(source)) > 1.0
        or float(np.min(candidate)) < 0.0
        or float(np.max(candidate)) > 1.0
        or not np.isfinite(margin)
        or margin <= 0.0
        or margin >= 0.5
    ):
        raise ROMMRec2020RenderError("source-anchored residual input is invalid")
    source64 = source.astype(np.float64)
    residual = candidate.astype(np.float64) - source64
    lower = np.minimum(source64, margin)
    upper = np.maximum(source64, 1.0 - margin)
    scale = np.ones(source.shape[:2], dtype=np.float64)
    for channel in range(3):
        value = residual[..., channel]
        positive = (value > 0.0) & (source64[..., channel] + value > upper[..., channel])
        negative = (value < 0.0) & (source64[..., channel] + value < lower[..., channel])
        bound = np.ones_like(value)
        bound[positive] = (
            upper[..., channel][positive] - source64[..., channel][positive]
        ) / value[positive]
        bound[negative] = (
            lower[..., channel][negative] - source64[..., channel][negative]
        ) / value[negative]
        scale = np.minimum(scale, bound)
    scale = np.clip(scale, 0.0, 1.0)
    output = np.asarray(source64 + scale[..., None] * residual, dtype=np.float32)
    if not bool(np.all(np.isfinite(output)) and np.all(output >= 0.0) and np.all(output <= 1.0)):
        raise ROMMRec2020RenderError("source-anchored residual output is not bounded")
    return output, np.asarray(scale, dtype=np.float32)


def render_official_romm_velvia_rec2020(
    input_path: Path,
    output_path: Path,
    *,
    profile_path: Path,
    root: Path,
) -> dict[str, Any]:
    """Render the evidence-bound Velvia look to relative Rec.2020 RGB16 PNG."""

    output_path = Path(output_path)
    if output_path.exists() or output_path.suffix.casefold() != ".png":
        raise ROMMRec2020RenderError("output must be a create-only .png path")
    profile, profile_sha256 = load_profile(profile_path, root=root)
    if profile["profile_id"] != PROFILE_ID:
        raise ROMMRec2020RenderError("official ROMM renderer requires its exact profile")
    mapped, mapping = load_and_map_official_romm_rgb16(input_path)
    return _render_mapped_velvia_rec2020(
        mapped,
        mapping,
        output_path,
        profile=profile,
        profile_sha256=profile_sha256,
        receipt_schema=RECEIPT_SCHEMA,
        root=root,
    )


def render_supported_prophoto_velvia_rec2020(
    input_path: Path,
    output_path: Path,
    *,
    profile_path: Path,
    root: Path,
) -> dict[str, Any]:
    """Render the exact allowlisted ProPhoto profile through the qualified look."""

    output_path = Path(output_path)
    if output_path.exists() or output_path.suffix.casefold() != ".png":
        raise ROMMRec2020RenderError("output must be a create-only .png path")
    profile, profile_sha256 = load_profile(profile_path, root=root)
    if profile["profile_id"] != PROPHOTO_PROFILE_ID:
        raise ROMMRec2020RenderError("supported ProPhoto renderer requires its exact profile")
    mapped, mapping = load_and_map_supported_prophoto_rgb16(
        input_path,
        allowed_profile_sha256s=tuple(profile["input"]["embedded_icc_sha256s"]),
    )
    return _render_mapped_velvia_rec2020(
        mapped,
        mapping,
        output_path,
        profile=profile,
        profile_sha256=profile_sha256,
        receipt_schema=PROPHOTO_RECEIPT_SCHEMA,
        root=root,
    )


def _render_mapped_velvia_rec2020(
    mapped: WorkingImage,
    mapping: dict[str, Any],
    output_path: Path,
    *,
    profile: dict[str, Any],
    profile_sha256: str,
    receipt_schema: str,
    root: Path,
) -> dict[str, Any]:
    assets = {binding["role"]: root / binding["path"] for binding in profile["assets"]}
    stats = json.loads(assets["style_statistics"].read_text(encoding="utf-8"))
    guard_payload = json.loads(assets["color_guardrails"].read_text(encoding="utf-8"))
    style_id = profile["style"]["id"]
    guard = dict(guard_payload["defaults"])
    guard.update(guard_payload["styles"].get(style_id, {}))
    style = profile["style"]
    candidate = apply_rec2020_safe_lab(
        mapped,
        destination_mean=np.asarray(stats["styles"][style_id]["mean"], dtype=np.float32),
        destination_std=np.asarray(stats["styles"][style_id]["std"], dtype=np.float32),
        style=style_id,
        strength=float(style["strength"]),
        luma_strength=float(style["luma_strength"]),
        gamut_mode=style["gamut_mode"],
        tone_rolloff=float(style["tone_rolloff"]),
        shadow_floor_l=float(style["shadow_floor_l"]),
        highlight_ceiling_l=float(style["highlight_ceiling_l"]),
        preserve_luma_detail_strength=float(style["preserve_luma_detail_strength"]),
        chroma_curve_strength=float(style["chroma_curve_strength"]),
        neutral_protect=float(guard["neutral_protect"]),
        skin_protect=float(guard["skin_protect"]),
        max_chroma_gain=guard.get("max_chroma_gain"),
        max_chroma_boost=guard.get("max_chroma_boost"),
        max_chroma_absolute=guard.get("max_chroma_absolute"),
    )
    margin = float(profile["residual_execution"]["rgb16_margin"])
    output_pixels, residual_scale = _source_anchored_interior_residual(
        mapped.pixels, candidate.pixels, margin=margin
    )
    output_working = replace(
        candidate,
        pixels=output_pixels,
        warnings=[
            *candidate.warnings,
            DecodeWarning(
                "source_anchored_rec2020_residual",
                "Applied source-anchored RGB16-interior residual execution.",
            ),
        ],
    )
    try:
        save_rec2020_16_png(output_working, output_path)
        stored = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
        expected = np.rint(
            linear_rec2020_to_rec2020(output_pixels) * 65535.0
        ).astype(np.uint16)
        if (
            stored is None
            or stored.dtype != np.uint16
            or stored.shape != expected.shape
            or not np.array_equal(stored[..., ::-1], expected)
        ):
            raise ROMMRec2020RenderError("rendered PNG exact sample readback failed")
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    return {
        "schema": receipt_schema,
        "profile_id": profile["profile_id"],
        "profile_sha256": profile_sha256,
        "input": mapping["input"],
        "ingress": {
            "transform": mapping["transform"],
            "diagnostics": mapping["diagnostics"],
        },
        "look": {
            "style": style_id,
            "gamut_mode": style["gamut_mode"],
            "median_residual_scale": float(np.median(residual_scale)),
            "fraction_residual_scale_below_0p5": float(np.mean(residual_scale < 0.5)),
        },
        "output": {
            "sha256": _sha256(output_path),
            "format": "PNG",
            "bit_depth": 16,
            "working_space": "linear_rec2020",
            "transfer": "BT.2020 SDR",
            "cicp": REC2020_SDR_CICP.hex(),
            "exact_sample_readback": True,
        },
        "output_claim": "film-inspired-look-approximation",
        "production_default_changed": False,
        "claim_ceiling": profile["claim_ceiling"],
    }


__all__ = [
    "PROFILE_ID",
    "PROFILE_SCHEMA",
    "PROPHOTO_PROFILE_ID",
    "PROPHOTO_PROFILE_SCHEMA",
    "PROPHOTO_RECEIPT_SCHEMA",
    "RECEIPT_SCHEMA",
    "ROMMRec2020RenderError",
    "load_profile",
    "render_official_romm_velvia_rec2020",
    "render_supported_prophoto_velvia_rec2020",
]
