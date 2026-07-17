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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, format="PNG", optimize=True)
    payload = bytearray(input_path.read_bytes())
    input_hash = hashlib.sha256(payload).hexdigest()
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
