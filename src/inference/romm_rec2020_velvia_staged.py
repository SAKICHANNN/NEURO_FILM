"""Disk-staged exact ProPhoto-to-Rec.2020 Velvia rendering."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

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


class _OutputWriter(Protocol):
    def write_rows(self, row_start: int, samples: np.ndarray) -> None: ...

    def finish(self) -> str: ...

    def abort(self) -> None: ...


def render_supported_prophoto_velvia_rec2020_staged(
    input_path: Path,
    output_path: Path,
    *,
    profile_path: Path,
    root: Path,
    scratch_dir: Path,
    row_chunk: int = 128,
    _ingress_mapper: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    _style_mapper: Callable[..., np.ndarray] | None = None,
    _gamut_mapper: Callable[
        [np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
    ]
    | None = None,
    _postcolor_mapper: Callable[
        [np.ndarray, np.ndarray, float], tuple[np.ndarray, np.ndarray]
    ]
    | None = None,
    _output_writer_factory: Callable[[Path, int, int], _OutputWriter] | None = None,
    _in_memory_staging: bool = False,
    _preprocess_workers: int = 1,
    _spill_mapped_for_context: bool = False,
) -> dict[str, Any]:
    """Render the qualified look with disk-staged, bounded-row intermediates."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    scratch_dir = Path(scratch_dir)
    if output_path.exists() or output_path.suffix.casefold() != ".png":
        raise ROMMRec2020RenderError("output must be a create-only .png path")
    if isinstance(row_chunk, bool) or not isinstance(row_chunk, int) or row_chunk <= 0:
        raise ROMMRec2020RenderError("row_chunk must be a positive integer")
    if (
        isinstance(_preprocess_workers, bool)
        or not isinstance(_preprocess_workers, int)
        or _preprocess_workers <= 0
        or _preprocess_workers > 8
    ):
        raise ROMMRec2020RenderError("preprocess_workers must be an integer in [1, 8]")
    ingress_mapper = _ingress_mapper or analytical_oklab_interior_rec2020
    style_mapper = _style_mapper or apply_safe_lab_transform
    profile, profile_sha256 = load_profile(profile_path, root=root)
    if profile["profile_id"] != PROPHOTO_PROFILE_ID:
        raise ROMMRec2020RenderError("staged renderer requires the supported ProPhoto profile")
    allowed_profiles = tuple(profile["input"]["embedded_icc_sha256s"])

    scratch_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="u1_4c19_", dir=scratch_dir, ignore_cleanup_errors=True
    ) as temporary:
        temporary_path = Path(temporary)
        direct_encoded_memmap = False
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
            if _in_memory_staging:
                if page.is_memmappable:
                    encoded = tifffile.memmap(input_path, page=0, mode="r")
                    direct_encoded_memmap = True
                else:
                    encoded = page.asarray()
            else:
                encoded = np.memmap(
                    temporary_path / "encoded.rgb16",
                    mode="w+",
                    dtype=np.uint16,
                    shape=shape,
                )
                page.asarray(out=encoded)

        height, width, _ = shape
        mapped_path = temporary_path / "mapped.f32"
        mapped = (
            np.empty(shape, dtype=np.float32)
            if _in_memory_staging and not _spill_mapped_for_context
            else np.memmap(
                mapped_path,
                mode="w+",
                dtype=np.float32,
                shape=shape,
            )
        )
        lab = (
            None
            if _in_memory_staging
            else np.memmap(
                temporary_path / "lab.f32",
                mode="w+",
                dtype=np.float32,
                shape=shape,
            )
        )
        mapped_count = 0
        minimum_chroma_scale = 1.0
        output_minimum = float("inf")
        output_maximum = float("-inf")
        ranges = [
            (y0, min(height, y0 + row_chunk))
            for y0 in range(0, height, row_chunk)
        ]

        def process_ingress(bounds: tuple[int, int]) -> tuple[int, float, float, float]:
            y0, y1 = bounds
            decoded = decode_prophoto_rgb16_to_linear_rec2020(
                np.asarray(encoded[y0:y1]), embedded_profile
            )
            source_in_gamut = np.all((decoded >= 0.0) & (decoded <= 1.0), axis=2)
            mapped_tile, chroma_scale = ingress_mapper(decoded)
            mapped[y0:y1] = mapped_tile
            if lab is not None:
                lab[y0:y1] = linear_rgb_to_lab(
                    mapped_tile, working_space="linear_rec2020"
                )
            return (
                int(np.count_nonzero(~source_in_gamut)),
                float(np.min(chroma_scale)),
                float(np.min(mapped_tile)),
                float(np.max(mapped_tile)),
            )

        if _preprocess_workers == 1:
            ingress_results = map(process_ingress, ranges)
        else:
            with ThreadPoolExecutor(max_workers=_preprocess_workers) as executor:
                ingress_results = tuple(executor.map(process_ingress, ranges))
        for count, minimum_scale, minimum_value, maximum_value in ingress_results:
            mapped_count += count
            minimum_chroma_scale = min(minimum_chroma_scale, minimum_scale)
            output_minimum = min(output_minimum, minimum_value)
            output_maximum = max(output_maximum, maximum_value)
        if _in_memory_staging:
            if direct_encoded_memmap:
                encoded._mmap.close()
            del encoded
            lab = np.empty(shape, dtype=np.float32)

            def process_lab(bounds: tuple[int, int]) -> None:
                y0, y1 = bounds
                assert lab is not None
                lab[y0:y1] = linear_rgb_to_lab(
                    mapped[y0:y1], working_space="linear_rec2020"
                )

            if _preprocess_workers == 1:
                for bounds in ranges:
                    process_lab(bounds)
            else:
                with ThreadPoolExecutor(max_workers=_preprocess_workers) as executor:
                    tuple(executor.map(process_lab, ranges))
        else:
            mapped.flush()
            assert lab is not None
            lab.flush()
            del encoded

        assert lab is not None
        if _spill_mapped_for_context:
            if not _in_memory_staging or not isinstance(mapped, np.memmap):
                raise ROMMRec2020RenderError("mapped context spill requires in-memory staging")
            mapped.flush()
            mapped._mmap.close()
            del mapped
            context = safe_lab_context_from_lab(lab)
            mapped = np.memmap(mapped_path, mode="r", dtype=np.float32, shape=shape)
        else:
            context = safe_lab_context_from_lab(lab)
        assets = {binding["role"]: root / binding["path"] for binding in profile["assets"]}
        stats = json.loads(assets["style_statistics"].read_text(encoding="utf-8"))
        guard_payload = json.loads(assets["color_guardrails"].read_text(encoding="utf-8"))
        style = profile["style"]
        style_id = style["id"]
        guard = dict(guard_payload["defaults"])
        guard.update(guard_payload["styles"].get(style_id, {}))
        output_samples = (
            None
            if _output_writer_factory is not None
            else np.memmap(
                temporary_path / "output.rgb16",
                mode="w+",
                dtype=np.uint16,
                shape=shape,
            )
        )
        output_writer = (
            _output_writer_factory(output_path, width, height)
            if _output_writer_factory is not None
            else None
        )
        output_sample_digest = hashlib.sha256()
        residual_scale = (
            np.empty((height, width), dtype=np.float32)
            if _in_memory_staging
            else np.memmap(
                temporary_path / "residual_scale.f32",
                mode="w+",
                dtype=np.float32,
                shape=(height, width),
            )
        )
        halo = 5
        margin = float(profile["residual_execution"]["rgb16_margin"])
        for y0 in range(0, height, row_chunk):
            y1 = min(height, y0 + row_chunk)
            expanded_y0 = max(0, y0 - halo)
            expanded_y1 = min(height, y1 + halo)
            source_lab = np.asarray(lab[expanded_y0:expanded_y1])
            styled_lab = style_mapper(
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
            if _gamut_mapper is None:
                output_lab = compress_source_to_working_gamut(
                    source_lab,
                    styled_lab,
                    working_space="linear_rec2020",
                )
                candidate = lab_to_linear_rgb(
                    output_lab, working_space="linear_rec2020"
                )
            else:
                output_lab, candidate = _gamut_mapper(source_lab, styled_lab)
            if (
                not np.isfinite(candidate).all()
                or np.any(candidate < -2e-6)
                or np.any(candidate > 1.0 + 2e-6)
            ):
                raise ROMMRec2020RenderError(
                    "staged safe-Lab candidate violates the working gamut"
                )
            candidate = np.asarray(np.clip(candidate, 0.0, 1.0), dtype=np.float32)
            crop0 = y0 - expanded_y0
            crop1 = crop0 + (y1 - y0)
            if _postcolor_mapper is None:
                output_pixels, scale = _source_anchored_interior_residual(
                    np.asarray(mapped[y0:y1]), candidate[crop0:crop1], margin=margin
                )
                encoded_output = linear_rec2020_to_rec2020(output_pixels)
                samples = np.rint(encoded_output * 65535.0).astype(np.uint16)
            else:
                samples, scale = _postcolor_mapper(
                    np.asarray(mapped[y0:y1]), candidate[crop0:crop1], margin
                )
                if (
                    samples.dtype != np.uint16
                    or samples.shape != (y1 - y0, width, 3)
                    or not samples.flags.c_contiguous
                ):
                    raise ROMMRec2020RenderError("postcolor mapper output is invalid")
                output_pixels = encoded_output = None
            output_sample_digest.update(np.ascontiguousarray(samples).tobytes())
            if output_writer is None:
                assert output_samples is not None
                output_samples[y0:y1] = samples
            else:
                output_writer.write_rows(y0, samples)
            residual_scale[y0:y1] = scale
        if output_samples is not None:
            output_samples.flush()
        else:
            assert output_writer is not None
            output_writer.finish()
        if not _in_memory_staging:
            residual_scale.flush()
        median_residual_scale = float(np.median(residual_scale))
        fraction_residual_scale_below_0p5 = float(
            np.mean(residual_scale < 0.5)
        )
        del (
            source_lab,
            styled_lab,
            output_lab,
            candidate,
            output_pixels,
            scale,
            encoded_output,
        )
        if not _in_memory_staging:
            for staged in (mapped, lab, residual_scale):
                staged.flush()
                staged._mmap.close()
        elif _spill_mapped_for_context:
            mapped._mmap.close()
        if output_samples is not None:
            save_rec2020_rgb16_png_samples(output_samples, output_path)
        stored = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
        if (
            stored is None
            or stored.dtype != np.uint16
            or stored.shape != shape
        ):
            output_path.unlink(missing_ok=True)
            raise ROMMRec2020RenderError("staged PNG exact sample readback failed")
        stored_rgb = np.ascontiguousarray(stored[..., ::-1])
        if hashlib.sha256(stored_rgb.tobytes()).digest() != output_sample_digest.digest():
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
                "median_residual_scale": median_residual_scale,
                "fraction_residual_scale_below_0p5": fraction_residual_scale_below_0p5,
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
        del stored
        if output_samples is not None:
            output_samples.flush()
            output_samples._mmap.close()
        return receipt




__all__ = ["render_supported_prophoto_velvia_rec2020_staged"]
