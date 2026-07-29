"""U6.P5G photographic severe/OOD stress for the P5F interimage operator."""

from __future__ import annotations

from functools import partial
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.eval.density_witness_frontier import linear_srgb_to_encoded
from src.eval.physical_spatial_photographic_stress import (
    _flat_region_p99,
    _isolated_excursions,
    _load_source,
    _preview,
)
from src.eval.physical_spatial_response import _profile
from src.eval.sensitometry_primitive import build_operator
from src.film_physics import (
    apply_bounded_development_adjacency,
    apply_dye_diffusion,
    apply_forward_scatter,
    apply_scanner_mtf,
    apply_spatial_response_pipeline,
    density_to_scan_transmittance,
    required_spatial_response_halo,
)
from src.film_physics.interimage_adjacency import (
    InterimageAdjacencyProfile,
    apply_interimage_adjacency,
    interimage_adjacency_profile_from_contract,
    required_interimage_adjacency_halo,
)


SCHEMA = "neuro_film.u6_p5g_interimage_photographic_stress_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P5G contract")
    return payload


def _combined_pipeline(
    exposure: np.ndarray,
    profile: Any,
    operator: Any,
    p5c_apply: Any,
    p5f_profile: InterimageAdjacencyProfile,
) -> np.ndarray:
    scattered = apply_forward_scatter(exposure, profile)
    density = p5c_apply(operator.apply(scattered), profile)
    density = apply_interimage_adjacency(density, p5f_profile)
    density = apply_dye_diffusion(density, profile)
    scan = density_to_scan_transmittance(density)
    return apply_scanner_mtf(scan, profile)


def _combined_pipeline_row_tiled(
    exposure: np.ndarray,
    profile: Any,
    operator: Any,
    p5c_apply: Any,
    p5f_profile: InterimageAdjacencyProfile,
    *,
    tile_rows: int,
) -> np.ndarray:
    if isinstance(tile_rows, bool) or not isinstance(tile_rows, int) or tile_rows <= 0:
        raise ValueError("tile_rows must be a positive integer")
    halo = required_spatial_response_halo(
        profile
    ) + required_interimage_adjacency_halo(p5f_profile)
    output = np.empty_like(exposure, dtype=np.float64)
    for y0 in range(0, exposure.shape[0], tile_rows):
        y1 = min(exposure.shape[0], y0 + tile_rows)
        source_y0 = max(0, y0 - halo)
        source_y1 = min(exposure.shape[0], y1 + halo)
        rendered = _combined_pipeline(
            exposure[source_y0:source_y1],
            profile,
            operator,
            p5c_apply,
            p5f_profile,
        )
        output[y0:y1] = rendered[y0 - source_y0 : y1 - source_y0]
    return output


def _linear_image(linear: np.ndarray) -> Image.Image:
    encoded = linear_srgb_to_encoded(np.clip(linear, 0.0, 1.0))
    return Image.fromarray(
        np.rint(encoded * 255.0).astype(np.uint8),
        mode="RGB",
    )


def _save_overview(
    rows: list[dict[str, Any]],
    fixed_ids: list[str],
    path: Path,
) -> str:
    by_id = {row["id"]: row for row in rows}
    tile_width, tile_height, header = 240, 160, 22
    canvas = Image.new(
        "RGB",
        (4 * tile_width, len(fixed_ids) * (tile_height + header)),
        color=(24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, sample_id in enumerate(fixed_ids):
        row = by_id[sample_id]
        y = row_index * (tile_height + header)
        draw.text((4, y + 4), sample_id, fill=(235, 235, 235))
        views = (
            row["source"],
            row["baseline"],
            row["candidate"],
            np.clip(0.5 + 32.0 * row["difference"], 0.0, 1.0),
        )
        for column, values in enumerate(views):
            image = _preview(values, (tile_width, tile_height))
            x = column * tile_width + (tile_width - image.width) // 2
            canvas.paste(image, (x, y + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=False, compress_level=9)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _crop_bounds(
    shape: tuple[int, int],
    centre: tuple[int, int],
    size: int,
) -> tuple[slice, slice]:
    height, width = shape
    crop_height = min(size, height)
    crop_width = min(size, width)
    y0 = min(max(centre[0] - crop_height // 2, 0), height - crop_height)
    x0 = min(max(centre[1] - crop_width // 2, 0), width - crop_width)
    return slice(y0, y0 + crop_height), slice(x0, x0 + crop_width)


def _save_worst_patches(
    rows: list[dict[str, Any]],
    fixed_ids: list[str],
    size: int,
    path: Path,
) -> str:
    by_id = {row["id"]: row for row in rows}
    header = 22
    canvas = Image.new(
        "RGB",
        (4 * size, len(fixed_ids) * (size + header)),
        color=(24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, sample_id in enumerate(fixed_ids):
        row = by_id[sample_id]
        y = row_index * (size + header)
        draw.text((4, y + 4), sample_id, fill=(235, 235, 235))
        magnitude = np.max(np.abs(row["difference"]), axis=-1)
        centre = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
        ys, xs = _crop_bounds(magnitude.shape, centre, size)
        views = (
            row["source"][ys, xs],
            row["baseline"][ys, xs],
            row["candidate"][ys, xs],
            np.clip(0.5 + 32.0 * row["difference"][ys, xs], 0.0, 1.0),
        )
        for column, values in enumerate(views):
            image = _linear_image(values)
            if image.size != (size, size):
                image = ImageOps.pad(
                    image,
                    (size, size),
                    method=Image.Resampling.NEAREST,
                    color=(24, 24, 24),
                    centering=(0.5, 0.5),
                )
            canvas.paste(image, (column * size, y + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=False, compress_level=9)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_interimage_photographic_stress(
    contract: dict[str, Any],
    manifest: list[dict[str, Any]],
    p5d_contract: dict[str, Any],
    p5c_contract: dict[str, Any],
    p5f_contract: dict[str, Any],
    p5a_contract: dict[str, Any],
    sensitometry_config: dict[str, Any],
    *,
    root: Path,
    overview_path: Path,
    worst_patches_path: Path,
) -> dict[str, Any]:
    expected = contract["input"]
    if len(manifest) != int(expected["expected_rows"]):
        raise ValueError("photographic manifest row count drift")
    if len({row["make"] for row in manifest}) != int(
        expected["expected_camera_makes"]
    ):
        raise ValueError("photographic camera-make count drift")

    p5d_input = p5d_contract["input"]
    for row in manifest:
        if (
            row["allowed_use"] != p5d_input["required_allowed_use"]
            or row["rights_scope"] != p5d_input["required_rights_scope"]
            or row["decoded_color_state"] != p5d_input["decoded_color_state"]
        ):
            raise ValueError("photographic input contract drift")

    profile = _profile(p5a_contract)
    p5c_candidate = p5c_contract["candidate"]
    p5c_apply = partial(
        apply_bounded_development_adjacency,
        maximum_absolute_transmittance_delta=float(
            p5c_candidate["maximum_absolute_transmittance_delta"]
        ),
        maximum_absolute_density_delta=float(
            p5c_candidate["maximum_absolute_density_delta"]
        ),
    )
    p5f_profile = interimage_adjacency_profile_from_contract(p5f_contract)
    operator = build_operator(sensitometry_config)
    partitions = [int(value) for value in contract["pipeline"]["row_partitions"]]
    gates = contract["automatic_gates"]
    rows: list[dict[str, Any]] = []
    visuals: list[dict[str, Any]] = []
    for source_row in manifest:
        source, input_sha = _load_source(source_row, root)
        baseline = apply_spatial_response_pipeline(
            source,
            profile,
            sensitometry_apply=operator.apply,
            adjacency_apply=p5c_apply,
        )
        candidate = _combined_pipeline(
            source,
            profile,
            operator,
            p5c_apply,
            p5f_profile,
        )
        partition_exact = {
            str(tile_rows): np.array_equal(
                candidate,
                _combined_pipeline_row_tiled(
                    source,
                    profile,
                    operator,
                    p5c_apply,
                    p5f_profile,
                    tile_rows=tile_rows,
                ),
            )
            for tile_rows in partitions
        }
        difference = candidate - baseline
        absolute = np.abs(difference)
        new_boundary = (
            ((candidate <= 0.0) | (candidate >= 1.0))
            & ~((baseline <= 0.0) | (baseline >= 1.0))
        )
        row = {
            "id": source_row["id"],
            "make": source_row["make"],
            "input_sha256": input_sha,
            "shape": list(source.shape),
            "finite_domain": bool(
                np.all(np.isfinite(candidate))
                and np.all(candidate > 0.0)
                and np.all(candidate <= 1.0)
            ),
            "partition_exact": partition_exact,
            "candidate_vs_baseline_max_abs": float(np.max(absolute)),
            "candidate_vs_baseline_p95_abs": float(
                np.quantile(absolute, 0.95)
            ),
            "flat_region_p99_abs": _flat_region_p99(source, difference),
            "maximum_difference_channel_spread": float(
                np.max(np.max(difference, axis=-1) - np.min(difference, axis=-1))
            ),
            "isolated_excursion_count": _isolated_excursions(
                difference,
                threshold=float(gates["isolated_excursion_threshold"]),
                radius=int(gates["isolated_support_radius_pixels"]),
                minimum_support=int(gates["minimum_isolated_support_count"]),
            ),
            "new_hard_boundary_fraction": float(np.mean(new_boundary)),
        }
        rows.append(row)
        if source_row["id"] in contract["visual_protocol"]["fixed_ids"]:
            visuals.append(
                {
                    "id": source_row["id"],
                    "source": source,
                    "baseline": baseline,
                    "candidate": candidate,
                    "difference": difference,
                }
            )

    fixed_ids = contract["visual_protocol"]["fixed_ids"]
    if sorted(row["id"] for row in visuals) != sorted(fixed_ids):
        raise ValueError("fixed visual IDs are incomplete")
    overview_sha = _save_overview(visuals, fixed_ids, overview_path)
    patch_sha = _save_worst_patches(
        visuals,
        fixed_ids,
        int(contract["visual_protocol"]["worst_patch_size"]),
        worst_patches_path,
    )
    population_p95 = float(
        np.quantile(
            [row["candidate_vs_baseline_p95_abs"] for row in rows],
            0.95,
        )
    )
    decisions = {
        "input_hashes": all(
            row["input_sha256"] == source["decoded_sha256"]
            for row, source in zip(rows, manifest, strict=True)
        ),
        "finite_domain": all(row["finite_domain"] for row in rows),
        "partition_exact": all(
            all(row["partition_exact"].values()) for row in rows
        ),
        "output_bound": max(
            row["candidate_vs_baseline_max_abs"] for row in rows
        )
        <= float(gates["maximum_candidate_vs_baseline_abs"]),
        "nontrivial_population": population_p95
        >= float(gates["minimum_population_p95_candidate_vs_baseline_abs"]),
        "flat_regions": max(row["flat_region_p99_abs"] for row in rows)
        <= float(gates["maximum_flat_region_p99_abs"]),
        "channel_spread": max(
            row["maximum_difference_channel_spread"] for row in rows
        )
        <= float(gates["maximum_difference_channel_spread"]),
        "isolated_excursions": sum(
            row["isolated_excursion_count"] for row in rows
        )
        <= int(gates["maximum_isolated_excursion_count"]),
        "new_boundaries": max(
            row["new_hard_boundary_fraction"] for row in rows
        )
        <= float(gates["maximum_new_hard_boundary_fraction"]),
    }
    passed = all(decisions.values())
    core = {
        "schema": (
            "neuro_film.u6_p5g_interimage_photographic_stress_report.v1"
        ),
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "rows": rows,
        "population_p95_candidate_vs_baseline_abs": population_p95,
        "overview_sha256": overview_sha,
        "worst_patches_sha256": patch_sha,
        "decisions": decisions,
        "automatic_pass": passed,
        "branch": contract["branch_rule"]["pass" if passed else "fail"],
    }
    evidence_id = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return {**core, "stable_evidence_id": evidence_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "_combined_pipeline",
    "_combined_pipeline_row_tiled",
    "evaluate_interimage_photographic_stress",
    "load_contract",
    "write_report",
]
