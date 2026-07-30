"""Fixed-profile photographic stress for generic creative optical diffusion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import uniform_filter

from src.eval.density_witness_frontier import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.filmfx.creative_diffusion import (
    CreativeDiffusionProfile,
    apply_creative_diffusion_linear,
)


SCHEMA = "neuro_film.u6_4b_creative_diffusion_photographic_stress.v1"


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.4B contract")
    return payload


def load_source(row: dict[str, Any], root: Path) -> tuple[np.ndarray, str]:
    path = root / row["decoded_path"]
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != row["decoded_sha256"]:
        raise ValueError(f"decoded input hash drift for {row['id']}")
    with Image.open(path) as image:
        encoded = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    return encoded_srgb_to_linear(encoded).astype(np.float32), digest


def isolated_excursions(
    difference: np.ndarray,
    *,
    threshold: float,
    radius: int,
    minimum_support: int,
) -> int:
    excursion = np.max(np.abs(difference), axis=-1) > threshold
    width = 2 * radius + 1
    support = uniform_filter(
        excursion.astype(np.float64),
        size=width,
        mode="constant",
        cval=0.0,
    ) * float(width * width)
    return int(np.count_nonzero(excursion & (support < minimum_support)))


def flat_region_p99(source: np.ndarray, difference: np.ndarray) -> float:
    luma = source @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    gradient = np.zeros_like(luma)
    gradient[:, 1:] += np.abs(np.diff(luma, axis=1))
    gradient[1:, :] += np.abs(np.diff(luma, axis=0))
    flat = gradient <= float(np.quantile(gradient, 0.25))
    values = np.max(np.abs(difference), axis=-1)[flat]
    return float(np.quantile(values, 0.99)) if values.size else 0.0


def encode_preview(linear: np.ndarray) -> Image.Image:
    encoded = linear_srgb_to_encoded(np.clip(linear.astype(np.float64), 0.0, 1.0))
    return Image.fromarray(
        np.rint(np.clip(encoded, 0.0, 1.0) * 255.0).astype(np.uint8),
        mode="RGB",
    )


def save_contact_sheet(
    visual_rows: list[dict[str, Any]],
    fixed_ids: list[str],
    path: Path,
) -> str:
    by_id = {str(row["id"]): row for row in visual_rows}
    tile_width, tile_height, header = 300, 200, 22
    canvas = Image.new(
        "RGB",
        (3 * tile_width, len(fixed_ids) * (tile_height + header)),
        color=(24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, sample_id in enumerate(fixed_ids):
        row = by_id[sample_id]
        y = row_index * (tile_height + header)
        draw.text((4, y + 4), sample_id, fill=(235, 235, 235))
        views = (
            encode_preview(row["source"]),
            encode_preview(row["output"]),
            encode_preview(np.clip(0.5 + 8.0 * row["difference"], 0.0, 1.0)),
        )
        for column, image in enumerate(views):
            image = ImageOps.contain(
                image,
                (tile_width, tile_height),
                method=Image.Resampling.LANCZOS,
            )
            x = column * tile_width + (tile_width - image.width) // 2
            canvas.paste(image, (x, y + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=False, compress_level=9)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(
    contract: dict[str, Any],
    manifest: list[dict[str, Any]],
    *,
    root: Path,
    output_root: Path,
) -> dict[str, Any]:
    expected = contract["input"]
    if len(manifest) != int(expected["expected_rows"]):
        raise ValueError("photographic manifest row count drift")
    if len({row["make"] for row in manifest}) != int(
        expected["expected_camera_makes"]
    ):
        raise ValueError("photographic camera-make count drift")
    for row in manifest:
        if row["allowed_use"] != expected["required_allowed_use"]:
            raise ValueError("allowed-use drift")
        if row["rights_scope"] != expected["required_rights_scope"]:
            raise ValueError("rights-scope drift")
        if row["decoded_color_state"] != expected["required_color_state"]:
            raise ValueError("decoded color-state drift")

    operator = contract["operator"]
    profile = CreativeDiffusionProfile(
        profile_id=str(operator["profile_id"]),
        sigmas_px=tuple(float(value) for value in operator["sigmas_px"]),
        scatter_fractions=tuple(
            float(value) for value in operator["scatter_fractions"]
        ),
    )
    gates = contract["automatic_gates"]
    fixed_ids = [str(value) for value in contract["visual_protocol"]["fixed_ids"]]
    rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)

    for source_row in manifest:
        source, source_sha = load_source(source_row, root)
        output = apply_creative_diffusion_linear(source, profile)
        repeated = apply_creative_diffusion_linear(source, profile)
        difference = output - source
        source_min = source.min(axis=(0, 1))
        source_max = source.max(axis=(0, 1))
        lower_escape = np.maximum(source_min - output.min(axis=(0, 1)), 0.0)
        upper_escape = np.maximum(output.max(axis=(0, 1)) - source_max, 0.0)
        bound_escape = float(max(lower_escape.max(), upper_escape.max()))
        source_luma = source @ np.asarray(
            [0.2126, 0.7152, 0.0722], dtype=np.float32
        )
        output_luma = output @ np.asarray(
            [0.2126, 0.7152, 0.0722], dtype=np.float32
        )
        new_boundary = (
            ((output <= 0.0) | (output >= 1.0))
            & ~((source <= 0.0) | (source >= 1.0))
        )
        encoded = encode_preview(output)
        output_path = output_root / f"{source_row['id']}__strong.png"
        encoded.save(output_path, format="PNG", optimize=False, compress_level=9)
        output_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
        row = {
            "id": source_row["id"],
            "make": source_row["make"],
            "source_sha256": source_sha,
            "output_sha256": output_sha,
            "shape": list(source.shape),
            "repeat_byte_exact": output.tobytes() == repeated.tobytes(),
            "component_bound_escape": bound_escape,
            "new_hard_boundary_fraction": float(np.mean(new_boundary)),
            "luma_mean_abs_drift": float(
                abs(float(output_luma.mean()) - float(source_luma.mean()))
            ),
            "maximum_abs_change": float(np.max(np.abs(difference))),
            "mean_abs_change": float(np.mean(np.abs(difference))),
            "flat_region_p99_abs_change": flat_region_p99(source, difference),
            "isolated_excursion_count": isolated_excursions(
                difference,
                threshold=float(gates["isolated_excursion_threshold"]),
                radius=int(gates["isolated_support_radius_pixels"]),
                minimum_support=int(gates["minimum_isolated_support_count"]),
            ),
        }
        rows.append(row)
        if source_row["id"] in fixed_ids:
            visual_rows.append(
                {
                    "id": source_row["id"],
                    "source": source,
                    "output": output,
                    "difference": difference,
                }
            )

    aggregates = {
        "source_count": len(rows),
        "camera_make_count": len({row["make"] for row in rows}),
        "worst_component_bound_escape": max(
            row["component_bound_escape"] for row in rows
        ),
        "worst_new_hard_boundary_fraction": max(
            row["new_hard_boundary_fraction"] for row in rows
        ),
        "worst_luma_mean_abs_drift": max(
            row["luma_mean_abs_drift"] for row in rows
        ),
        "worst_maximum_abs_change": max(
            row["maximum_abs_change"] for row in rows
        ),
        "median_mean_abs_change": float(
            np.median([row["mean_abs_change"] for row in rows])
        ),
        "worst_flat_region_p99_abs_change": max(
            row["flat_region_p99_abs_change"] for row in rows
        ),
        "total_isolated_excursion_count": sum(
            row["isolated_excursion_count"] for row in rows
        ),
        "all_repeat_byte_exact": all(
            row["repeat_byte_exact"] for row in rows
        ),
    }
    checks = {
        "component_bounds": aggregates["worst_component_bound_escape"]
        <= float(gates["maximum_component_bound_escape"]),
        "new_boundaries": aggregates["worst_new_hard_boundary_fraction"]
        <= float(gates["maximum_new_hard_boundary_fraction"]),
        "luma_mean": aggregates["worst_luma_mean_abs_drift"]
        <= float(gates["maximum_population_luma_mean_abs_drift"]),
        "maximum_change": aggregates["worst_maximum_abs_change"]
        <= float(gates["maximum_per_image_abs_change"]),
        "minimum_visible_change": aggregates["median_mean_abs_change"]
        >= float(gates["minimum_population_median_mean_abs_change"]),
        "flat_regions": aggregates["worst_flat_region_p99_abs_change"]
        <= float(gates["maximum_population_flat_region_p99_abs_change"]),
        "isolated_excursions": aggregates["total_isolated_excursion_count"]
        <= int(gates["maximum_isolated_excursion_count"]),
        "repeat": aggregates["all_repeat_byte_exact"]
        is bool(gates["repeat_outputs_byte_exact"]),
    }
    automatic_pass = all(checks.values())
    contact_sheet_path = output_root / "contact_sheet.png"
    contact_sheet_sha = (
        save_contact_sheet(visual_rows, fixed_ids, contact_sheet_path)
        if automatic_pass
        else None
    )
    report: dict[str, Any] = {
        "schema": "neuro_film.u6_4b_creative_diffusion_photographic_stress.report.v1",
        "experiment_id": contract["experiment_id"],
        "rows": rows,
        "aggregates": aggregates,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "contact_sheet_sha256": contact_sheet_sha,
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(canonical_bytes(report)).hexdigest()
    return report


__all__ = [
    "SCHEMA",
    "canonical_bytes",
    "evaluate",
    "flat_region_p99",
    "isolated_excursions",
    "load_contract",
    "load_source",
]
