"""Held-out photographic and partition stress for the U6.P5C spatial chain."""

from __future__ import annotations

from functools import partial
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
from src.eval.physical_spatial_response import _profile
from src.eval.sensitometry_primitive import build_operator
from src.film_physics import (
    SpatialResponseProfile,
    apply_bounded_development_adjacency,
    apply_development_adjacency,
    apply_dye_diffusion,
    apply_forward_scatter,
    apply_scanner_mtf,
    apply_spatial_response_pipeline,
    apply_spatial_response_pipeline_row_tiled,
    density_to_scan_transmittance,
)


SCHEMA = "neuro_film.u6_p5d_spatial_photographic_stress_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P5D contract")
    return payload


def _zero_adjacency(profile: SpatialResponseProfile) -> SpatialResponseProfile:
    return SpatialResponseProfile(
        pixel_pitch_um=profile.pixel_pitch_um,
        forward_scatter_sigma_um_rgb=profile.forward_scatter_sigma_um_rgb,
        development_adjacency_sigma_um_rgb=(
            profile.development_adjacency_sigma_um_rgb
        ),
        development_adjacency_gain_rgb=(0.0, 0.0, 0.0),
        dye_diffusion_sigma_um_rgb=profile.dye_diffusion_sigma_um_rgb,
        scanner_mtf_sigma_um_rgb=profile.scanner_mtf_sigma_um_rgb,
        gaussian_truncate=profile.gaussian_truncate,
    )


def _load_source(row: dict[str, Any], root: Path) -> tuple[np.ndarray, str]:
    path = root / row["decoded_path"]
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != row["decoded_sha256"]:
        raise ValueError(f"decoded input hash drift for {row['id']}")
    with Image.open(path) as image:
        encoded = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    return encoded_srgb_to_linear(encoded), digest


def _flat_region_p99(source: np.ndarray, difference: np.ndarray) -> float:
    luma = source @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    gradient = np.zeros_like(luma)
    gradient[:, 1:] += np.abs(np.diff(luma, axis=1))
    gradient[1:, :] += np.abs(np.diff(luma, axis=0))
    threshold = float(np.quantile(gradient, 0.25))
    flat = gradient <= threshold
    values = np.max(np.abs(difference), axis=-1)[flat]
    return float(np.quantile(values, 0.99)) if values.size else 0.0


def _isolated_excursions(
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


def _wrong_order(
    exposure: np.ndarray,
    profile: SpatialResponseProfile,
    operator: Any,
    adjacency_apply: Any,
) -> np.ndarray:
    scattered = apply_forward_scatter(exposure, profile)
    density = adjacency_apply(operator.apply(scattered), profile)
    density = apply_dye_diffusion(density, profile)
    scale = max(float(np.max(density)), 1.0)
    scanner_first = apply_scanner_mtf(density / scale, profile) * scale
    return density_to_scan_transmittance(scanner_first)


def _preview(linear: np.ndarray, size: tuple[int, int]) -> Image.Image:
    encoded = linear_srgb_to_encoded(np.clip(linear, 0.0, 1.0))
    image = Image.fromarray(
        np.rint(np.clip(encoded, 0.0, 1.0) * 255.0).astype(np.uint8),
        mode="RGB",
    )
    return ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)


def _save_contact_sheet(
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
            row["zero"],
            row["bounded"],
            np.clip(0.5 + 16.0 * row["difference"], 0.0, 1.0),
        )
        for column, values in enumerate(views):
            image = _preview(values, (tile_width, tile_height))
            x = column * tile_width + (tile_width - image.width) // 2
            canvas.paste(image, (x, y + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=False, compress_level=9)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_photographic_stress(
    contract: dict[str, Any],
    manifest: list[dict[str, Any]],
    p5c_contract: dict[str, Any],
    p5a_contract: dict[str, Any],
    sensitometry_config: dict[str, Any],
    *,
    root: Path,
    contact_sheet_path: Path,
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
            raise ValueError("photographic allowed-use drift")
        if row["rights_scope"] != expected["required_rights_scope"]:
            raise ValueError("photographic rights-scope drift")
        if row["decoded_color_state"] != expected["decoded_color_state"]:
            raise ValueError("photographic color-state drift")

    profile = _profile(p5a_contract)
    zero_profile = _zero_adjacency(profile)
    candidate = p5c_contract["candidate"]
    bounded_apply = partial(
        apply_bounded_development_adjacency,
        maximum_absolute_transmittance_delta=float(
            candidate["maximum_absolute_transmittance_delta"]
        ),
        maximum_absolute_density_delta=float(
            candidate["maximum_absolute_density_delta"]
        ),
    )
    operator = build_operator(sensitometry_config)
    tile_rows = [int(value) for value in contract["pipeline"]["row_partitions"]]
    gates = contract["automatic_gates"]
    rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    for source_row in manifest:
        source, input_sha = _load_source(source_row, root)
        zero = apply_spatial_response_pipeline(
            source,
            zero_profile,
            sensitometry_apply=operator.apply,
        )
        bounded = apply_spatial_response_pipeline(
            source,
            profile,
            sensitometry_apply=operator.apply,
            adjacency_apply=bounded_apply,
        )
        partition_exact: dict[str, bool] = {}
        for tile_size in tile_rows:
            tiled = apply_spatial_response_pipeline_row_tiled(
                source,
                profile,
                sensitometry_apply=operator.apply,
                adjacency_apply=bounded_apply,
                tile_rows=tile_size,
            )
            partition_exact[str(tile_size)] = np.array_equal(bounded, tiled)
        wrong = _wrong_order(source, profile, operator, bounded_apply)
        difference = bounded - zero
        absolute = np.abs(difference)
        new_boundary = (
            ((bounded <= 0.0) | (bounded >= 1.0))
            & ~((zero <= 0.0) | (zero >= 1.0))
        )
        row = {
            "id": source_row["id"],
            "make": source_row["make"],
            "input_sha256": input_sha,
            "shape": list(source.shape),
            "finite_domain": bool(
                np.all(np.isfinite(bounded))
                and np.all(bounded > 0.0)
                and np.all(bounded <= 1.0)
            ),
            "partition_exact": partition_exact,
            "bounded_vs_zero_max_abs": float(np.max(absolute)),
            "bounded_vs_zero_p95_abs": float(np.quantile(absolute, 0.95)),
            "flat_region_p99_abs": _flat_region_p99(source, difference),
            "isolated_excursion_count": _isolated_excursions(
                difference,
                threshold=float(gates["isolated_excursion_threshold"]),
                radius=int(gates["isolated_support_radius_pixels"]),
                minimum_support=int(gates["minimum_isolated_support_count"]),
            ),
            "new_hard_boundary_fraction": float(np.mean(new_boundary)),
            "correct_vs_wrong_order_max_abs": float(
                np.max(np.abs(bounded - wrong))
            ),
        }
        rows.append(row)
        if source_row["id"] in contract["visual_protocol"]["fixed_ids"]:
            visual_rows.append(
                {
                    "id": source_row["id"],
                    "source": source,
                    "zero": zero,
                    "bounded": bounded,
                    "difference": difference,
                }
            )
    visual_ids = [row["id"] for row in visual_rows]
    if sorted(visual_ids) != sorted(contract["visual_protocol"]["fixed_ids"]):
        raise ValueError("fixed visual IDs are incomplete")
    contact_sheet_sha = _save_contact_sheet(
        visual_rows,
        contract["visual_protocol"]["fixed_ids"],
        contact_sheet_path,
    )
    population_p95 = float(
        np.quantile([row["bounded_vs_zero_p95_abs"] for row in rows], 0.95)
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
        "adjacency_bound": max(
            row["bounded_vs_zero_max_abs"] for row in rows
        )
        <= float(gates["maximum_bounded_vs_zero_adjacency_abs"]) + 1e-12,
        "nontrivial_population": population_p95
        >= float(
            gates["minimum_population_p95_bounded_vs_zero_adjacency_abs"]
        ),
        "flat_regions": max(row["flat_region_p99_abs"] for row in rows)
        <= float(gates["maximum_flat_region_p99_adjacency_abs"]),
        "isolated_excursions": sum(
            row["isolated_excursion_count"] for row in rows
        )
        <= int(gates["maximum_isolated_adjacency_excursion_count"]),
        "new_boundaries": max(
            row["new_hard_boundary_fraction"] for row in rows
        )
        <= float(gates["maximum_new_hard_boundary_fraction"]),
        "wrong_order": min(
            row["correct_vs_wrong_order_max_abs"] for row in rows
        )
        > 1e-4,
    }
    passed = all(decisions.values())
    core = {
        "schema": "neuro_film.u6_p5d_spatial_photographic_stress_report.v1",
        "node": contract["node"],
        "claim_ceiling": expected["domain_use"],
        "rows": rows,
        "population_p95_bounded_vs_zero_adjacency_abs": population_p95,
        "contact_sheet_sha256": contact_sheet_sha,
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
