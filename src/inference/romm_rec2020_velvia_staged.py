"""Disk-staged exact ProPhoto-to-Rec.2020 Velvia rendering."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile

from src.color_engine.gamut import compress_source_to_working_gamut
from src.color_engine.lab import lab_to_linear_rgb, linear_rgb_to_lab
from src.color_engine.oklab_analytical_interior import analytical_oklab_interior_rec2020
from src.color_engine.safe_lab import (
    apply_safe_lab_transform,
    safe_lab_context_from_lab,
)
from src.preprocess import (
    REC2020_SDR_CICP,
    linear_rec2020_to_rec2020,
    save_rec2020_rgb16_png_samples,
)
from src.preprocess.prophoto_icc import decode_prophoto_rgb16_to_linear_rec2020

from .romm_rec2020_velvia import (
    PROPHOTO_PROFILE_ID,
    PROPHOTO_RECEIPT_SCHEMA,
    ROMMRec2020RenderError,
    _sha256,
    _source_anchored_interior_residual,
    load_profile,
)


def render_supported_prophoto_velvia_rec2020_staged(
    input_path: Path,
    output_path: Path,
    *,
    profile_path: Path,
    root: Path,
    scratch_dir: Path,
    row_chunk: int = 128,
) -> dict[str, Any]:
    """Render the qualified look with disk-staged, bounded-row intermediates."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    scratch_dir = Path(scratch_dir)
    if output_path.exists() or output_path.suffix.casefold() != ".png":
        raise ROMMRec2020RenderError("output must be a create-only .png path")
    if isinstance(row_chunk, bool) or not isinstance(row_chunk, int) or row_chunk <= 0:
        raise ROMMRec2020RenderError("row_chunk must be a positive integer")
    profile, profile_sha256 = load_profile(profile_path, root=root)
    if profile["profile_id"] != PROPHOTO_PROFILE_ID:
        raise ROMMRec2020RenderError("staged renderer requires the supported ProPhoto profile")
    allowed_profiles = tuple(profile["input"]["embedded_icc_sha256s"])

    scratch_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="u1_4c19_", dir=scratch_dir) as temporary:
        temporary_path = Path(temporary)
        with tifffile.TiffFile(input_path) as document:
            if len(document.pages) != 1:
                raise ROMMRec2020RenderError("staged input must contain one TIFF page")
            page = document.pages[0]
            profile_tag = page.tags.get(34675)
            embedded_profile = bytes(profile_tag.value) if profile_tag is not None else b""
            embedded_sha256 = hashlib.sha256(embedded_profile).hexdigest()
            orientation = int(page.tags.get("Orientation").value) if page.tags.get("Orientation") else 1
            if embedded_sha256 not in allowed_profiles or orientation != 1:
                raise ROMMRec2020RenderError("staged input profile or orientation is unsupported")
            shape = tuple(int(value) for value in page.shape)
            if page.dtype != np.dtype(np.uint16) or len(shape) != 3 or shape[2] != 3:
                raise ROMMRec2020RenderError("staged input must be RGB16 TIFF")
            encoded = np.memmap(
                temporary_path / "encoded.rgb16",
                mode="w+",
                dtype=np.uint16,
                shape=shape,
            )
            page.asarray(out=encoded)

        height, width, _ = shape
        mapped = np.memmap(
            temporary_path / "mapped.f32",
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )
        lab = np.memmap(
            temporary_path / "lab.f32",
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )
        mapped_count = 0
        minimum_chroma_scale = 1.0
        output_minimum = float("inf")
        output_maximum = float("-inf")
        for y0 in range(0, height, row_chunk):
            y1 = min(height, y0 + row_chunk)
            decoded = decode_prophoto_rgb16_to_linear_rec2020(
                np.asarray(encoded[y0:y1]), embedded_profile
            )
            source_in_gamut = np.all((decoded >= 0.0) & (decoded <= 1.0), axis=2)
            mapped_tile, chroma_scale = analytical_oklab_interior_rec2020(decoded)
            mapped[y0:y1] = mapped_tile
            lab[y0:y1] = linear_rgb_to_lab(
                mapped_tile, working_space="linear_rec2020"
            )
            mapped_count += int(np.count_nonzero(~source_in_gamut))
            minimum_chroma_scale = min(minimum_chroma_scale, float(np.min(chroma_scale)))
            output_minimum = min(output_minimum, float(np.min(mapped_tile)))
            output_maximum = max(output_maximum, float(np.max(mapped_tile)))
        mapped.flush()
        lab.flush()
        del encoded

        context = safe_lab_context_from_lab(lab)
        assets = {binding["role"]: root / binding["path"] for binding in profile["assets"]}
        stats = json.loads(assets["style_statistics"].read_text(encoding="utf-8"))
        guard_payload = json.loads(assets["color_guardrails"].read_text(encoding="utf-8"))
        style = profile["style"]
        style_id = style["id"]
        guard = dict(guard_payload["defaults"])
        guard.update(guard_payload["styles"].get(style_id, {}))
        output_samples = np.memmap(
            temporary_path / "output.rgb16",
            mode="w+",
            dtype=np.uint16,
            shape=shape,
        )
        residual_scale = np.memmap(
            temporary_path / "residual_scale.f32",
            mode="w+",
            dtype=np.float32,
            shape=(height, width),
        )
        halo = 5
        margin = float(profile["residual_execution"]["rgb16_margin"])
        for y0 in range(0, height, row_chunk):
            y1 = min(height, y0 + row_chunk)
            expanded_y0 = max(0, y0 - halo)
            expanded_y1 = min(height, y1 + halo)
            source_lab = np.asarray(lab[expanded_y0:expanded_y1])
            styled_lab = apply_safe_lab_transform(
                source_lab,
                source_context=context,
                destination_mean=np.asarray(stats["styles"][style_id]["mean"], dtype=np.float32),
                destination_std=np.asarray(stats["styles"][style_id]["std"], dtype=np.float32),
                style=style_id,
                strength=float(style["strength"]),
                luma_strength=float(style["luma_strength"]),
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
            output_lab = compress_source_to_working_gamut(
                source_lab,
                styled_lab,
                working_space="linear_rec2020",
            )
            candidate = lab_to_linear_rgb(output_lab, working_space="linear_rec2020")
            crop0 = y0 - expanded_y0
            crop1 = crop0 + (y1 - y0)
            output_pixels, scale = _source_anchored_interior_residual(
                np.asarray(mapped[y0:y1]), candidate[crop0:crop1], margin=margin
            )
            encoded_output = linear_rec2020_to_rec2020(output_pixels)
            output_samples[y0:y1] = np.rint(encoded_output * 65535.0).astype(np.uint16)
            residual_scale[y0:y1] = scale
        output_samples.flush()
        residual_scale.flush()
        save_rec2020_rgb16_png_samples(output_samples, output_path)
        stored = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
        if (
            stored is None
            or stored.dtype != np.uint16
            or stored.shape != shape
            or not np.array_equal(stored[..., ::-1], output_samples)
        ):
            output_path.unlink(missing_ok=True)
            raise ROMMRec2020RenderError("staged PNG exact sample readback failed")

        receipt = {
            "schema": PROPHOTO_RECEIPT_SCHEMA,
            "profile_id": profile["profile_id"],
            "profile_sha256": profile_sha256,
            "input": {
                "sha256": _sha256(input_path),
                "embedded_icc_sha256": embedded_sha256,
                "width": width,
                "height": height,
                "bit_depth": 16,
            },
            "ingress": {
                "transform": {
                    "mapper_id": "oklab-oog-only-soft-interval-maximum-chroma-v1",
                    "softness": 1.0 / 64.0,
                    "rgb16_margin": 2.0 / 65535.0,
                    "bisection_iterations": 24,
                    "hard_clipping": False,
                    "posthoc_limiting": False,
                },
                "diagnostics": {
                    "mapped_pixel_count": mapped_count,
                    "mapped_pixel_fraction": float(mapped_count / (height * width)),
                    "minimum_chroma_scale": minimum_chroma_scale,
                    "output_minimum": output_minimum,
                    "output_maximum": output_maximum,
                },
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
        del (
            source_lab,
            styled_lab,
            output_lab,
            candidate,
            output_pixels,
            scale,
            encoded_output,
            stored,
            mapped_tile,
            chroma_scale,
            decoded,
        )
        for staged in (mapped, lab, output_samples, residual_scale):
            staged.flush()
            staged._mmap.close()
        return receipt




__all__ = ["render_supported_prophoto_velvia_rec2020_staged"]

