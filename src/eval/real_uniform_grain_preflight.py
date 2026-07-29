"""Integrity, decode and fixed-crop preflight for uniform film-grain TIFFs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tifffile

from src.eval.real_uniform_grain_nps import fixed_fractional_crops
from src.eval.real_uniform_grain_source import hash_file, validate_contract


class UniformGrainPreflightError(RuntimeError):
    """Raised when a frozen TIFF is not safe for the NPS feasibility audit."""


def _basic_crop_metrics(values: np.ndarray) -> dict[str, float]:
    normalized = values.astype(np.float64) / float(np.iinfo(values.dtype).max)
    return {
        "mean": float(np.mean(normalized)),
        "standard_deviation": float(np.std(normalized)),
        "minimum": float(np.min(normalized)),
        "maximum": float(np.max(normalized)),
        "boundary_fraction": float(
            np.mean((normalized <= 0.0) | (normalized >= 1.0))
        ),
    }


def inspect_uniform_grain_tiff(
    *,
    path: Path,
    expected: dict[str, Any],
    crop_size: int,
    centers_yx: list[list[float]],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Read one exact TIFF and keep RGB distinct from infrared alpha."""
    if (
        path.stat().st_size != int(expected["expected_bytes"])
        or hash_file(path, "sha1") != str(expected["api_sha1"])
    ):
        raise UniformGrainPreflightError("TIFF identity mismatch")
    with tifffile.TiffFile(path) as tiff:
        if len(tiff.pages) != 1:
            raise UniformGrainPreflightError("TIFF must contain one page")
        page = tiff.pages[0]
        image = page.asarray()
        compression = int(page.compression)
        resolution = tuple(
            float(value[0] / value[1])
            for value in (
                page.tags["XResolution"].value,
                page.tags["YResolution"].value,
            )
        )
    if (
        image.dtype != np.uint16
        or image.ndim != 3
        or image.shape[2] != 4
        or tuple(image.shape[:2])
        != (int(expected["height"]), int(expected["width"]))
    ):
        raise UniformGrainPreflightError(
            "TIFF must be exact uint16 RGBA at frozen dimensions"
        )
    if compression != 1:
        raise UniformGrainPreflightError("TIFF must remain uncompressed")
    rgb = np.ascontiguousarray(image[..., :3])
    infrared = np.ascontiguousarray(image[..., 3])
    del image
    rgb_mean = np.mean(rgb.astype(np.float64), axis=2)
    crop_metrics: list[dict[str, Any]] = []
    for crop_index, (rgb_crop, ir_crop) in enumerate(
        zip(
            fixed_fractional_crops(
                rgb_mean,
                crop_size=crop_size,
                centers_yx=centers_yx,
            ),
            fixed_fractional_crops(
                infrared,
                crop_size=crop_size,
                centers_yx=centers_yx,
            ),
            strict=True,
        )
    ):
        crop_metrics.append(
            {
                "crop_index": crop_index,
                "rgb_mean_code": _basic_crop_metrics(
                    np.rint(rgb_crop).astype(np.uint16)
                ),
                "infrared_code": _basic_crop_metrics(ir_crop),
            }
        )
    report = {
        "source_id": expected["title"][5:-4],
        "film_stock_id": expected["film_stock_id"],
        "path": str(expected["path"]),
        "bytes": path.stat().st_size,
        "sha1": expected["api_sha1"],
        "sha256": hash_file(path, "sha256"),
        "shape": list(rgb.shape[:2]),
        "dtype": str(rgb.dtype),
        "samples": 4,
        "compression": "none",
        "resolution_dpi": list(resolution),
        "rgb_ir_separated": True,
        "crop_metrics": crop_metrics,
    }
    return report, rgb, infrared


def _preview(values: np.ndarray, *, maximum_size: tuple[int, int]) -> Image.Image:
    code8 = np.right_shift(values, 8).astype(np.uint8)
    if code8.ndim == 2:
        image = Image.fromarray(code8, mode="L").convert("RGB")
    else:
        image = Image.fromarray(code8, mode="RGB")
    image.thumbnail(maximum_size, Image.Resampling.LANCZOS)
    return image


def run_preflight(
    *,
    root: Path,
    source_config: dict[str, Any],
    analysis_config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Verify all exact files and build a non-selective RGB/IR contact sheet."""
    validate_contract(source_config)
    if output_dir.exists():
        raise FileExistsError("uniform-grain preflight is create-only")
    output_dir.mkdir(parents=True)
    crop_size = int(analysis_config["pixel_contract"]["crop_size_pixels"])
    centers = analysis_config["pixel_contract"]["fixed_fractional_centers_yx"]
    rows: list[dict[str, Any]] = []
    previews: list[tuple[str, Image.Image, Image.Image]] = []
    for expected in sorted(
        source_config["files"],
        key=lambda row: str(row["title"]),
    ):
        report, rgb, infrared = inspect_uniform_grain_tiff(
            path=root / str(expected["path"]),
            expected=expected,
            crop_size=crop_size,
            centers_yx=centers,
        )
        rows.append(report)
        previews.append(
            (
                str(expected["title"])[5:-4],
                _preview(rgb, maximum_size=(720, 430)),
                _preview(infrared, maximum_size=(720, 430)),
            )
        )
        del rgb, infrared
    font = ImageFont.load_default()
    row_height = 468
    sheet = Image.new("RGB", (1480, row_height * len(previews) + 32), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (8, 8),
        "U6.P4R exact scanner-code RGB and infrared-alpha previews",
        fill="black",
        font=font,
    )
    for index, (source_id, rgb_preview, ir_preview) in enumerate(previews):
        y = 32 + index * row_height
        draw.text((8, y), f"{source_id} / RGB", fill="black", font=font)
        draw.text((748, y), f"{source_id} / IR alpha", fill="black", font=font)
        sheet.paste(rgb_preview, (8, y + 24))
        sheet.paste(ir_preview, (748, y + 24))
    contact_path = output_dir / "source_contact_sheet.png"
    sheet.save(contact_path, "PNG", compress_level=6)
    core = {
        "schema": "neuro_film.u6_p4r_uniform_grain_preflight_report.v1",
        "source_contract_id": source_config["experiment_id"],
        "source_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_exact_integrity": len(rows) == 8,
        "all_single_page_uint16_rgba": all(
            row["dtype"] == "uint16"
            and row["samples"] == 4
            and row["compression"] == "none"
            for row in rows
        ),
        "rgb_ir_separation_required": True,
        "fixed_crop_count_per_scan": len(centers),
        "contact_sheet_sha256": hash_file(contact_path, "sha256"),
        "rows": rows,
        "visual_review_required_before_nps": True,
        "nps_analysis_allowed": False,
        "claim_ceiling": source_config["claim_ceiling"],
    }
    stable_id = hashlib.sha256(
        json.dumps(
            core,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {**core, "stable_evidence_id": stable_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "UniformGrainPreflightError",
    "inspect_uniform_grain_tiff",
    "run_preflight",
    "write_report",
]
