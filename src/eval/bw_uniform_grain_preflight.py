"""Decode and fixed-crop preflight for U6.P4Z B&W uniform scans."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.eval.bw_uniform_grain_source import validate_contract
from src.eval.real_uniform_grain_source import hash_file


class BWUniformGrainPreflightError(RuntimeError):
    """Raised when a frozen B&W preflight input or invariant fails."""


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = root / binding["path"]
    if hash_file(path, "sha256") != binding["sha256"]:
        raise BWUniformGrainPreflightError(f"parent hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BWUniformGrainPreflightError("parent payload must be an object")
    return payload


def _fixed_crops(
    array: np.ndarray,
    *,
    crop_size: int,
    centers_yx: list[list[float]],
) -> list[np.ndarray]:
    height, width = array.shape
    if crop_size <= 0 or crop_size > min(height, width):
        raise BWUniformGrainPreflightError("invalid crop size")
    crops: list[np.ndarray] = []
    for fraction_y, fraction_x in centers_yx:
        center_y = int(round(float(fraction_y) * (height - 1)))
        center_x = int(round(float(fraction_x) * (width - 1)))
        y0 = min(max(center_y - crop_size // 2, 0), height - crop_size)
        x0 = min(max(center_x - crop_size // 2, 0), width - crop_size)
        crops.append(np.ascontiguousarray(array[y0 : y0 + crop_size, x0 : x0 + crop_size]))
    return crops


def _crop_summary(crop: np.ndarray) -> dict[str, float]:
    normalized = crop.astype(np.float64) / 65535.0
    return {
        "mean": float(np.mean(normalized)),
        "standard_deviation": float(np.std(normalized)),
        "p001": float(np.quantile(normalized, 0.001)),
        "p999": float(np.quantile(normalized, 0.999)),
    }


def _display_crop(crop: np.ndarray) -> Image.Image:
    low, high = np.quantile(crop.astype(np.float64), [0.005, 0.995])
    if not high > low:
        raise BWUniformGrainPreflightError("degenerate visual crop")
    encoded = np.clip((crop.astype(np.float64) - low) / (high - low), 0.0, 1.0)
    return Image.fromarray(np.rint(encoded * 255.0).astype(np.uint8), mode="L")


def run_preflight(
    *,
    root: Path,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], Image.Image]:
    """Run exact decode, profile, crop and bounded-uniformity checks."""
    if (
        contract.get("schema")
        != "neuro_film.u6_p4z1_bw_uniform_grain_preflight_contract.v1"
    ):
        raise BWUniformGrainPreflightError("unsupported preflight contract")
    if contract.get("operator_fitting_allowed") is not False:
        raise BWUniformGrainPreflightError("preflight cannot fit an operator")
    source = _load_bound(root, contract["parents"]["source_contract"])
    manifest = _load_bound(root, contract["parents"]["acquisition_manifest"])
    validate_contract(source)
    manifest_by_title = {row["title"]: row for row in manifest["rows"]}
    if (
        manifest.get("source_contract_id") != source["experiment_id"]
        or set(manifest_by_title) != {row["title"] for row in source["files"]}
    ):
        raise BWUniformGrainPreflightError("manifest/source identity mismatch")

    decode = contract["decode_contract"]
    crop_contract = contract["crop_contract"]
    gates = contract["automatic_gates"]
    rows: list[dict[str, Any]] = []
    display_rows: list[list[Image.Image]] = []
    for expected in sorted(source["files"], key=lambda row: row["title"]):
        path = root / expected["path"]
        acquired = manifest_by_title[expected["title"]]
        if (
            hash_file(path, "sha256") != acquired["sha256"]
            or path.stat().st_size != expected["expected_bytes"]
        ):
            raise BWUniformGrainPreflightError("local payload identity mismatch")
        with Image.open(path) as image:
            mode = image.mode
            size = image.size
            info = dict(image.info)
            array = np.asarray(image)
        icc = info.get("icc_profile")
        if not isinstance(icc, bytes):
            raise BWUniformGrainPreflightError("embedded ICC profile is absent")
        dpi = info.get("dpi")
        if (
            mode != decode["mode"]
            or str(array.dtype) != decode["dtype"]
            or array.ndim != 2
            or size != (expected["width"], expected["height"])
            or len(icc) != int(decode["icc_profile_bytes"])
            or hashlib.sha256(icc).hexdigest() != decode["icc_profile_sha256"]
            or not isinstance(dpi, tuple)
            or len(dpi) != 2
            or max(abs(float(value) - float(decode["expected_dpi"])) for value in dpi)
            > float(decode["maximum_dpi_absolute_error"])
        ):
            raise BWUniformGrainPreflightError("decode/profile contract mismatch")
        crops = _fixed_crops(
            array,
            crop_size=int(crop_contract["crop_size_pixels"]),
            centers_yx=crop_contract["fixed_fractional_centers_yx"],
        )
        summaries = [_crop_summary(crop) for crop in crops]
        standard_deviations = [row["standard_deviation"] for row in summaries]
        means = [row["mean"] for row in summaries]
        zero_fraction = float(np.mean(array == 0))
        full_fraction = float(np.mean(array == 65535))
        gate_results = {
            "zero_fraction": zero_fraction
            <= float(gates["maximum_exact_zero_fraction"]),
            "full_scale_fraction": full_fraction
            <= float(gates["maximum_exact_full_scale_fraction"]),
            "crop_standard_deviation_minimum": min(standard_deviations)
            >= float(gates["minimum_crop_standard_deviation"]),
            "crop_standard_deviation_maximum": max(standard_deviations)
            <= float(gates["maximum_crop_standard_deviation"]),
            "crop_mean_range": max(means) - min(means)
            <= float(gates["maximum_within_file_crop_mean_range"]),
        }
        rows.append(
            {
                "title": expected["title"],
                "film_stock_id": expected["film_stock_id"],
                "path": expected["path"],
                "sha256": acquired["sha256"],
                "width": int(size[0]),
                "height": int(size[1]),
                "mode": mode,
                "dtype": str(array.dtype),
                "icc_profile_sha256": hashlib.sha256(icc).hexdigest(),
                "dpi": [float(value) for value in dpi],
                "minimum_code": int(np.min(array)),
                "maximum_code": int(np.max(array)),
                "exact_zero_fraction": zero_fraction,
                "exact_full_scale_fraction": full_fraction,
                "crop_summaries": summaries,
                "within_file_crop_mean_range": float(max(means) - min(means)),
                "gate_results": gate_results,
                "automatic_pass": all(gate_results.values()),
            }
        )
        display_rows.append([_display_crop(crop) for crop in crops])

    crop_size = int(crop_contract["crop_size_pixels"])
    label_height = 24
    sheet = Image.new(
        "L",
        (3 * crop_size, len(display_rows) * (3 * crop_size + label_height)),
        color=0,
    )
    draw = ImageDraw.Draw(sheet)
    for file_index, images in enumerate(display_rows):
        row_y = file_index * (3 * crop_size + label_height)
        draw.text((4, row_y + 4), rows[file_index]["title"], fill=255)
        for index, image in enumerate(images):
            y = row_y + label_height + (index // 3) * crop_size
            x = (index % 3) * crop_size
            sheet.paste(image, (x, y))

    stable = {
        "schema": "neuro_film.u6_p4z1_bw_uniform_grain_preflight_report.v1",
        "contract_sha256": hash_file(
            root / "configs/u6_p4z1_bw_uniform_grain_preflight_v1.json",
            "sha256",
        ),
        "source_contract_sha256": contract["parents"]["source_contract"]["sha256"],
        "acquisition_manifest_sha256": contract["parents"][
            "acquisition_manifest"
        ]["sha256"],
        "source_count": len(rows),
        "rows": rows,
        "automatic_pass": all(row["automatic_pass"] for row in rows),
        "visual_review_permitted": all(row["automatic_pass"] for row in rows),
        "operator_fitting_executed": False,
        "decision": (
            "open_fixed_model_confirmation_after_visual_source_review"
            if all(row["automatic_pass"] for row in rows)
            else "close_bw_uniform_grain_source_before_visual_review"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_id = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**stable, "stable_evidence_id": stable_id}, sheet


__all__ = [
    "BWUniformGrainPreflightError",
    "run_preflight",
]
