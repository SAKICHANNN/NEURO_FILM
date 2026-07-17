"""Explicit non-generative synthetic failure operators for FilmStyleSafe A0.

These operators intentionally create controlled chromatic stress cases for
evaluation development. They never use generative RGB models.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class SyntheticFailureError(ValueError):
    """Raised when a synthetic failure operator contract fails closed."""


def canonical_parameter_hash(parameters: Mapping[str, Any]) -> str:
    encoded = (json.dumps(parameters, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def highlight_chroma_island_v0(
    image: Image.Image,
    *,
    seed: int,
    center_xy_norm: tuple[float, float],
    radius_norm: float,
    luma_threshold: float,
    chroma_boost: float,
) -> Image.Image:
    """Inject a deterministic high-chroma island into bright pixels.

    Parameters are normalized to image size. The operator is replayable and
    explicit: it only remaps existing RGB values inside a circular highlight
    mask.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    cx = float(center_xy_norm[0]) * (width - 1)
    cy = float(center_xy_norm[1]) * (height - 1)
    radius = float(radius_norm) * float(min(width, height))
    if radius <= 1.0:
        raise SyntheticFailureError("radius_norm too small")
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # Deterministic soft disk; seed only selects an optional subpixel jitter.
    rng = np.random.default_rng(seed)
    jitter = float(rng.uniform(-0.25, 0.25))
    mask = (dist <= (radius + jitter)) & (luma >= float(luma_threshold))
    if not np.any(mask):
        raise SyntheticFailureError("highlight chroma island mask is empty")
    out = rgb.copy()
    # Push red/magenta chroma while compressing green to form a neon island.
    out[mask, 0] = np.clip(out[mask, 0] * (1.0 + chroma_boost) + 0.08 * chroma_boost, 0.0, 1.0)
    out[mask, 1] = np.clip(out[mask, 1] * (1.0 - 0.55 * chroma_boost), 0.0, 1.0)
    out[mask, 2] = np.clip(out[mask, 2] * (1.0 + 0.35 * chroma_boost), 0.0, 1.0)
    return Image.fromarray(np.round(out * 255.0).astype(np.uint8), mode="RGB")


def _write_operator_png(
    *,
    input_path: Path,
    output_path: Path,
    parameters: Mapping[str, Any],
    result: Image.Image,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, format="PNG", optimize=True)
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    output_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    param_hash = canonical_parameter_hash(parameters)
    return {
        "operator_id": parameters["operator_id"],
        "input_path": str(input_path.as_posix()),
        "output_path": str(output_path.as_posix()),
        "input_hash": input_hash,
        "output_hash": output_hash,
        "parameter_hash": param_hash,
        "parameters": dict(parameters),
    }


def run_highlight_chroma_island_v0(
    input_path: Path,
    output_path: Path,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen v0 operator and write a PNG evidence file."""
    required = {
        "seed",
        "center_xy_norm",
        "radius_norm",
        "luma_threshold",
        "chroma_boost",
        "operator_id",
    }
    if set(parameters) < required:
        raise SyntheticFailureError(f"missing parameters: {sorted(required - set(parameters))}")
    if parameters["operator_id"] != "explicit-highlight-chroma-island-v0":
        raise SyntheticFailureError("unexpected operator_id")
    center = parameters["center_xy_norm"]
    if not (isinstance(center, list) and len(center) == 2):
        raise SyntheticFailureError("center_xy_norm must be [x, y]")
    with Image.open(input_path) as image:
        image.load()
        result = highlight_chroma_island_v0(
            image,
            seed=int(parameters["seed"]),
            center_xy_norm=(float(center[0]), float(center[1])),
            radius_norm=float(parameters["radius_norm"]),
            luma_threshold=float(parameters["luma_threshold"]),
            chroma_boost=float(parameters["chroma_boost"]),
        )
    return _write_operator_png(
        input_path=input_path,
        output_path=output_path,
        parameters=parameters,
        result=result,
    )


def highlight_chroma_speckle_v1(
    image: Image.Image,
    *,
    seed: int,
    center_xy_norm: tuple[float, float],
    radius_norm: float,
    luma_threshold: float,
    chroma_boost: float,
    speckle_density: float,
    blob_radius_px: int,
) -> Image.Image:
    """Inject sparse high-frequency chroma speckles into bright pixels.

    Unlike the solid v0 disk, this only remaps a seeded subset of highlight
    pixels (and optional tiny blobs), producing speckled chroma stress closer
    to ID11-style neon islands while remaining fully explicit and non-generative.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    if not (0.0 < float(speckle_density) <= 1.0):
        raise SyntheticFailureError("speckle_density must be in (0, 1]")
    if int(blob_radius_px) < 0:
        raise SyntheticFailureError("blob_radius_px must be >= 0")
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    cx = float(center_xy_norm[0]) * (width - 1)
    cy = float(center_xy_norm[1]) * (height - 1)
    radius = float(radius_norm) * float(min(width, height))
    if radius <= 1.0:
        raise SyntheticFailureError("radius_norm too small")
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    eligible = (dist <= radius) & (luma >= float(luma_threshold))
    eligible_idx = np.flatnonzero(eligible.reshape(-1))
    if eligible_idx.size == 0:
        raise SyntheticFailureError("highlight chroma speckle mask is empty")
    rng = np.random.default_rng(seed)
    n_seeds = max(1, int(round(float(speckle_density) * float(eligible_idx.size))))
    n_seeds = min(n_seeds, int(eligible_idx.size))
    chosen = rng.choice(eligible_idx, size=n_seeds, replace=False)
    mask = np.zeros((height, width), dtype=bool)
    blob = int(blob_radius_px)
    for flat in chosen:
        y0 = int(flat) // width
        x0 = int(flat) % width
        if blob == 0:
            mask[y0, x0] = True
            continue
        y1 = max(0, y0 - blob)
        y2 = min(height, y0 + blob + 1)
        x1 = max(0, x0 - blob)
        x2 = min(width, x0 + blob + 1)
        local = eligible[y1:y2, x1:x2]
        yy_l, xx_l = np.mgrid[y1:y2, x1:x2]
        disk = ((xx_l - x0) ** 2 + (yy_l - y0) ** 2) <= (blob * blob)
        mask[y1:y2, x1:x2] |= local & disk
    if not np.any(mask):
        raise SyntheticFailureError("highlight chroma speckle selection is empty")
    out = rgb.copy()
    out[mask, 0] = np.clip(out[mask, 0] * (1.0 + chroma_boost) + 0.10 * chroma_boost, 0.0, 1.0)
    out[mask, 1] = np.clip(out[mask, 1] * (1.0 - 0.70 * chroma_boost), 0.0, 1.0)
    out[mask, 2] = np.clip(out[mask, 2] * (1.0 + 0.25 * chroma_boost), 0.0, 1.0)
    return Image.fromarray(np.round(out * 255.0).astype(np.uint8), mode="RGB")


def run_highlight_chroma_speckle_v1(
    input_path: Path,
    output_path: Path,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen v1 speckle operator and write a PNG evidence file."""
    required = {
        "seed",
        "center_xy_norm",
        "radius_norm",
        "luma_threshold",
        "chroma_boost",
        "speckle_density",
        "blob_radius_px",
        "operator_id",
    }
    if set(parameters) < required:
        raise SyntheticFailureError(f"missing parameters: {sorted(required - set(parameters))}")
    if parameters["operator_id"] != "explicit-highlight-chroma-speckle-v1":
        raise SyntheticFailureError("unexpected operator_id")
    center = parameters["center_xy_norm"]
    if not (isinstance(center, list) and len(center) == 2):
        raise SyntheticFailureError("center_xy_norm must be [x, y]")
    with Image.open(input_path) as image:
        image.load()
        result = highlight_chroma_speckle_v1(
            image,
            seed=int(parameters["seed"]),
            center_xy_norm=(float(center[0]), float(center[1])),
            radius_norm=float(parameters["radius_norm"]),
            luma_threshold=float(parameters["luma_threshold"]),
            chroma_boost=float(parameters["chroma_boost"]),
            speckle_density=float(parameters["speckle_density"]),
            blob_radius_px=int(parameters["blob_radius_px"]),
        )
    return _write_operator_png(
        input_path=input_path,
        output_path=output_path,
        parameters=parameters,
        result=result,
    )


def bounded_bloom_halation_hardneg_v0(
    image: Image.Image,
    *,
    strength: float,
    threshold: float,
    edge_threshold: float,
    min_radius: float,
    max_radius: float,
    radius_gamma: float,
    scale_count: int,
) -> Image.Image:
    """Apply deterministic simple screen-halation as a legitimate-local hard negative.

    Intentional warm highlight bloom must remain distinguishable from neon chroma
    islands / ID11 speckles. Reuses the production ``halation_layer`` operator;
    no generative RGB rewrite.
    """
    from src.filmfx import composite_layers, halation_layer

    if image.mode != "RGB":
        image = image.convert("RGB")
    if not (0.0 < float(strength) <= 0.45):
        raise SyntheticFailureError("hard-negative strength must be in (0, 0.45]")
    if float(max_radius) <= float(min_radius):
        raise SyntheticFailureError("max_radius must exceed min_radius")
    base = np.asarray(image, dtype=np.float32) / 255.0
    layer = halation_layer(
        base,
        strength=float(strength),
        threshold=float(threshold),
        edge_threshold=float(edge_threshold),
        min_radius=float(min_radius),
        max_radius=float(max_radius),
        radius_gamma=float(radius_gamma),
        scale_count=int(scale_count),
        name="filmstylesafe_hardneg_halation_v0",
    )
    out = composite_layers(base, [layer], output_margin=0)
    encoded = np.round(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    source_u8 = np.round(np.clip(base, 0.0, 1.0) * 255.0).astype(np.uint8)
    if np.array_equal(encoded, source_u8):
        raise SyntheticFailureError("bounded bloom/halation hard-negative produced no visible change")
    return Image.fromarray(encoded, mode="RGB")


def run_bounded_bloom_halation_hardneg_v0(
    input_path: Path,
    output_path: Path,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen hard-negative operator and write a PNG evidence file."""
    required = {
        "operator_id",
        "hard_negative_label",
        "strength",
        "threshold",
        "edge_threshold",
        "min_radius",
        "max_radius",
        "radius_gamma",
        "scale_count",
    }
    if set(parameters) < required:
        raise SyntheticFailureError(f"missing parameters: {sorted(required - set(parameters))}")
    if parameters["operator_id"] != "explicit-bounded-bloom-halation-hardneg-v0":
        raise SyntheticFailureError("unexpected operator_id")
    if parameters["hard_negative_label"] != "bounded_bloom_or_halation":
        raise SyntheticFailureError("unexpected hard_negative_label")
    with Image.open(input_path) as image:
        image.load()
        result = bounded_bloom_halation_hardneg_v0(
            image,
            strength=float(parameters["strength"]),
            threshold=float(parameters["threshold"]),
            edge_threshold=float(parameters["edge_threshold"]),
            min_radius=float(parameters["min_radius"]),
            max_radius=float(parameters["max_radius"]),
            radius_gamma=float(parameters["radius_gamma"]),
            scale_count=int(parameters["scale_count"]),
        )
    return _write_operator_png(
        input_path=input_path,
        output_path=output_path,
        parameters=parameters,
        result=result,
    )
