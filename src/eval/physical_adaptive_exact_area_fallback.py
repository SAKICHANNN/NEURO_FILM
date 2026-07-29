"""U6.P4J adaptive exact-area fallback audit."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from src.eval.physical_density_conditioned_lod import (
    _area_mean,
    _display_noise_out_of_domain,
)
from src.eval.physical_density_conditioned_structure import (
    profiles_from_contract,
)
from src.eval.physical_two_cumulant_nonstationarity import (
    _base_scenes,
    _contrast_ratio,
    _island_metrics,
    _minimum_map_correlation,
    _region_metrics,
    _step_center,
)
from src.film_physics.density_conditioned_structure import (
    adaptive_exact_area_mask,
    compile_two_cumulant_profiles,
    render_density_conditioned_structure,
    render_density_conditioned_structure_adaptive_lod,
    render_density_conditioned_structure_adaptive_lod_region,
)


SCHEMA = "neuro_film.u6_p4j_adaptive_exact_area_fallback_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4j_adaptive_exact_area_fallback_report.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_exact(root: Path, path: str, expected: str) -> dict[str, Any]:
    absolute = root / path
    if _sha256(absolute) != expected:
        raise ValueError(f"U6.P4J parent drift: {path}")
    return json.loads(absolute.read_text(encoding="utf-8"))


def load_contract(
    root: Path, path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if _sha256(path) != expected_sha256:
        raise ValueError("U6.P4J contract hash mismatch")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != SCHEMA
        or contract["training_allowed"]
        or contract["photograph_access_allowed"]
        or contract["production_integration_allowed"]
        or contract["router"]["per_image_fit_or_normalization_allowed"]
        or contract["router"]["output_clipping_allowed"]
        or contract["router"]["sparse_product_executor_claim_allowed"]
    ):
        raise ValueError("unsupported U6.P4J contract")
    parents = contract["parents"]
    parent = _load_exact(
        root,
        parents["p4d_contract"],
        parents["p4d_contract_sha256"],
    )
    p4i = _load_exact(
        root,
        parents["p4i_contract"],
        parents["p4i_contract_sha256"],
    )
    decision = _load_exact(
        root,
        parents["p4i_decision"],
        parents["p4i_decision_sha256"],
    )
    if (
        decision["decision"]
        != "close_nonstationary_two_cumulant_integration_retain_p4h_stationary_only"
        or not decision["next_leaf"].startswith("U6.P4J")
    ):
        raise ValueError("U6.P4I did not open adaptive exact fallback")
    return contract, parent, p4i


def _evaluate_split(
    contract: dict[str, Any],
    parent: dict[str, Any],
    p4i: dict[str, Any],
    *,
    seed_offset: int,
) -> dict[str, Any]:
    scenes = _base_scenes(p4i)
    base_profiles = tuple(
        replace(profile, boundary_mode="normalized-support-v1")
        for profile in profiles_from_contract(parent, seed_offset=seed_offset)
    )
    threshold = float(contract["router"]["exact_fallback_threshold"])
    rows = []
    for factor in (int(value) for value in contract["lod_factors"]):
        compiled = compile_two_cumulant_profiles(
            base_profiles, pixel_size_factor=factor
        )
        for name, base_target in scenes.items():
            target = _area_mean(base_target, factor)
            reference = -np.log(
                _area_mean(
                    render_density_conditioned_structure(
                        base_target, base_profiles
                    ).transmittance,
                    factor,
                )
            )
            direct = render_density_conditioned_structure_adaptive_lod(
                target,
                base_profiles,
                pixel_size_factor=factor,
                density_range_threshold=threshold,
            )
            repeat = render_density_conditioned_structure_adaptive_lod(
                target,
                base_profiles,
                pixel_size_factor=factor,
                density_range_threshold=threshold,
            )
            assembled = np.empty_like(direct.density)
            for y0 in range(0, target.shape[0], 31):
                y1 = min(target.shape[0], y0 + 31)
                assembled[y0:y1] = (
                    render_density_conditioned_structure_adaptive_lod_region(
                        target,
                        base_profiles,
                        pixel_size_factor=factor,
                        density_range_threshold=threshold,
                        origin_yx=(y0, 0),
                        shape=(y1 - y0, target.shape[1]),
                    ).density
                )
            mask = adaptive_exact_area_mask(
                target,
                base_profiles,
                pixel_size_factor=factor,
                density_range_threshold=threshold,
            )
            boundary = binary_dilation(mask) ^ binary_erosion(mask)
            boundary_error = (
                float(
                    np.mean(
                        np.abs(
                            direct.density.astype(np.float64) - reference
                        )[boundary]
                    )
                )
                if np.any(boundary)
                else 0.0
            )
            constant_target = np.broadcast_to(
                np.mean(target, axis=(0, 1), keepdims=True),
                target.shape,
            ).copy()
            constant = render_density_conditioned_structure(
                constant_target, compiled
            )
            mean_error, minimum_variance, maximum_variance = (
                _region_metrics(
                    target, reference, direct.density.astype(np.float64)
                )
            )
            row: dict[str, Any] = {
                "scene": name,
                "factor": factor,
                "minimum_density_map_correlation": _minimum_map_correlation(
                    reference, direct.density
                ),
                "maximum_region_mean_absolute_difference": mean_error,
                "minimum_region_variance_ratio": minimum_variance,
                "maximum_region_variance_ratio": maximum_variance,
                "constant_rate_density_map_correlation": (
                    _minimum_map_correlation(target, constant.density)
                ),
                "display_noise_out_of_domain_fraction": (
                    _display_noise_out_of_domain(
                        direct.transmittance,
                        seed=62971 + seed_offset + factor,
                    )
                ),
                "exact_fallback_fraction": float(np.mean(mask)),
                "executor_boundary_mean_absolute_error": boundary_error,
                "minimum_density": float(np.min(direct.density)),
                "maximum_transmittance": float(
                    np.max(direct.transmittance)
                ),
                "repeat_exact": bool(
                    np.array_equal(direct.density, repeat.density)
                    and np.array_equal(
                        direct.transmittance, repeat.transmittance
                    )
                ),
                "row_partition_exact": bool(
                    np.array_equal(direct.density, assembled)
                ),
                "density_sha256": hashlib.sha256(
                    direct.density.tobytes(order="C")
                ).hexdigest(),
            }
            if name == "step":
                row["step_contrast_ratio"] = _contrast_ratio(
                    target, reference, direct.density
                )
                row["step_center_shift_pixels"] = abs(
                    _step_center(direct.density) - _step_center(reference)
                )
            elif name.startswith("checker_"):
                row["checkerboard_contrast_ratio"] = _contrast_ratio(
                    target, reference, direct.density
                )
            else:
                ratio, shift = _island_metrics(
                    target,
                    reference,
                    direct.density,
                    radius=float(p4i["scenes"]["islands"]["base_radius"])
                    / factor,
                )
                row["island_excess_ratio"] = ratio
                row["island_centroid_shift_pixels"] = shift
            rows.append(row)
    return {"rows": rows}


def evaluate_adaptive_exact_area_fallback(
    contract: dict[str, Any],
    parent: dict[str, Any],
    p4i: dict[str, Any],
) -> dict[str, Any]:
    scene = contract["scenes"]
    development = _evaluate_split(
        contract,
        parent,
        p4i,
        seed_offset=int(scene["development_seed_offset"]),
    )
    confirmation = _evaluate_split(
        contract,
        parent,
        p4i,
        seed_offset=int(scene["confirmation_seed_offset"]),
    )
    rows = development["rows"] + confirmation["rows"]
    gates = contract["automatic_gates"]
    step_rows = [row for row in rows if row["scene"] == "step"]
    checker_rows = [
        row for row in rows if row["scene"].startswith("checker_")
    ]
    island_rows = [row for row in rows if row["scene"] == "islands"]
    checks = {
        "density_map_correlation": min(
            row["minimum_density_map_correlation"] for row in rows
        )
        >= float(gates["minimum_density_map_correlation"]),
        "region_mean": max(
            row["maximum_region_mean_absolute_difference"] for row in rows
        )
        <= float(gates["maximum_region_mean_absolute_difference"]),
        "region_variance": min(
            row["minimum_region_variance_ratio"] for row in rows
        )
        >= float(gates["minimum_region_variance_ratio"])
        and max(row["maximum_region_variance_ratio"] for row in rows)
        <= float(gates["maximum_region_variance_ratio"]),
        "step_contrast": min(
            row["step_contrast_ratio"] for row in step_rows
        )
        >= float(gates["minimum_step_contrast_ratio"])
        and max(row["step_contrast_ratio"] for row in step_rows)
        <= float(gates["maximum_step_contrast_ratio"]),
        "step_center": max(
            row["step_center_shift_pixels"] for row in step_rows
        )
        <= float(gates["maximum_step_center_shift_pixels"]),
        "checkerboard_contrast": min(
            row["checkerboard_contrast_ratio"] for row in checker_rows
        )
        >= float(gates["minimum_checkerboard_contrast_ratio"])
        and max(
            row["checkerboard_contrast_ratio"] for row in checker_rows
        )
        <= float(gates["maximum_checkerboard_contrast_ratio"]),
        "island_excess": min(
            row["island_excess_ratio"] for row in island_rows
        )
        >= float(gates["minimum_island_excess_ratio"])
        and max(row["island_excess_ratio"] for row in island_rows)
        <= float(gates["maximum_island_excess_ratio"]),
        "island_centroid": max(
            row["island_centroid_shift_pixels"] for row in island_rows
        )
        <= float(gates["maximum_island_centroid_shift_pixels"]),
        "constant_rate_negative": max(
            abs(row["constant_rate_density_map_correlation"])
            for row in rows
        )
        <= float(gates["maximum_constant_rate_density_map_correlation"]),
        "display_noise_negative": min(
            row["display_noise_out_of_domain_fraction"] for row in rows
        )
        >= float(gates["minimum_display_noise_out_of_domain_fraction"]),
        "fallback_mean": float(
            np.mean([row["exact_fallback_fraction"] for row in rows])
        )
        <= float(gates["maximum_exact_fallback_fraction_mean"]),
        "fallback_step": max(
            row["exact_fallback_fraction"] for row in step_rows
        )
        <= float(gates["maximum_step_exact_fallback_fraction"]),
        "executor_boundary": max(
            row["executor_boundary_mean_absolute_error"] for row in rows
        )
        <= float(gates["maximum_executor_boundary_mean_absolute_error"]),
        "physical_domain": min(row["minimum_density"] for row in rows)
        >= 0.0
        and max(row["maximum_transmittance"] for row in rows) <= 1.0,
        "repeat_exact": all(row["repeat_exact"] for row in rows),
        "row_partition_exact": all(
            row["row_partition_exact"] for row in rows
        ),
    }
    passed = all(checks.values())
    return {
        "schema": REPORT_SCHEMA,
        "node": contract["node"],
        "development": development,
        "confirmation": confirmation,
        "checks": checks,
        "automatic_pass": passed,
        "decision": (
            contract["branch_rule"]["pass"]
            if passed
            else contract["branch_rule"]["fail"]
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "evaluate_adaptive_exact_area_fallback",
    "load_contract",
]
