"""U6.P4I nonstationary stress for two-cumulant film structure LOD."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_density_conditioned_lod import (
    _area_mean,
    _display_noise_out_of_domain,
)
from src.eval.physical_density_conditioned_structure import (
    profiles_from_contract,
)
from src.film_physics.density_conditioned_structure import (
    compile_two_cumulant_profiles,
    render_density_conditioned_structure,
    render_density_conditioned_structure_region,
)


SCHEMA = "neuro_film.u6_p4i_two_cumulant_nonstationarity_stress_contract.v1"
REPORT_SCHEMA = (
    "neuro_film.u6_p4i_two_cumulant_nonstationarity_stress_report.v1"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_exact(root: Path, path: str, expected: str) -> dict[str, Any]:
    absolute = root / path
    if _sha256(absolute) != expected:
        raise ValueError(f"U6.P4I parent drift: {path}")
    return json.loads(absolute.read_text(encoding="utf-8"))


def load_contract(
    root: Path, path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _sha256(path) != expected_sha256:
        raise ValueError("U6.P4I contract hash mismatch")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != SCHEMA
        or contract["training_allowed"]
        or contract["photograph_access_allowed"]
        or contract["production_integration_allowed"]
        or contract["compiler"]["per_image_fit_or_normalization_allowed"]
        or contract["compiler"]["empirical_parameter_fit_allowed"]
        or contract["compiler"]["output_clipping_allowed"]
    ):
        raise ValueError("unsupported U6.P4I contract")
    parents = contract["parents"]
    parent = _load_exact(
        root,
        parents["p4d_contract"],
        parents["p4d_contract_sha256"],
    )
    _load_exact(
        root,
        parents["p4h_contract"],
        parents["p4h_contract_sha256"],
    )
    decision = _load_exact(
        root,
        parents["p4h_decision"],
        parents["p4h_decision_sha256"],
    )
    if (
        decision["decision"]
        != "retain_two_cumulant_normalized_support_open_nonstationarity_stress"
        or not decision["next_leaf"].startswith("U6.P4I")
    ):
        raise ValueError("U6.P4H did not open nonstationarity stress")
    return contract, parent


def _base_scenes(contract: dict[str, Any]) -> dict[str, np.ndarray]:
    spec = contract["scenes"]
    coarse_height, coarse_width = (
        int(value) for value in spec["coarse_shape_at_factor_4"]
    )
    height, width = coarse_height * 4, coarse_width * 4
    scales = np.asarray(spec["channel_density_scales"], dtype=np.float64)

    step_spec = spec["step"]
    step = np.full(
        (height, width),
        float(step_spec["low_density"]),
        dtype=np.float64,
    )
    step[:, width // 2 :] = float(step_spec["high_density"])

    scenes = {"step": step}
    checker = spec["checkerboard"]
    ys, xs = np.indices((height, width))
    for cell in (int(value) for value in checker["base_cell_sizes"]):
        select = ((ys // cell) + (xs // cell)) % 2
        scenes[f"checker_{cell}"] = np.where(
            select == 0,
            float(checker["low_density"]),
            float(checker["high_density"]),
        )

    island_spec = spec["islands"]
    islands = np.full(
        (height, width),
        float(island_spec["background_density"]),
        dtype=np.float64,
    )
    radius = int(island_spec["base_radius"])
    high_center = (height // 2, width * 3 // 10)
    low_center = (height // 2, width * 7 // 10)
    high_mask = (
        (ys - high_center[0]) ** 2 + (xs - high_center[1]) ** 2
        <= radius * radius
    )
    low_mask = (
        (ys - low_center[0]) ** 2 + (xs - low_center[1]) ** 2
        <= radius * radius
    )
    islands[high_mask] = float(island_spec["high_density"])
    islands[low_mask] = float(island_spec["low_density"])
    scenes["islands"] = islands
    return {
        name: np.clip(
            values[..., None] * scales[None, None, :], 0.0, 2.0
        )
        for name, values in scenes.items()
    }


def _constant_masks(target: np.ndarray) -> list[np.ndarray]:
    masks = []
    rounded = np.round(target[..., 0], decimals=8)
    for value in np.unique(rounded):
        mask = rounded == value
        if np.count_nonzero(mask) >= 64:
            masks.append(mask)
    return masks


def _region_metrics(
    target: np.ndarray, reference: np.ndarray, direct: np.ndarray
) -> tuple[float, float, float]:
    maximum_mean = 0.0
    minimum_variance = float("inf")
    maximum_variance = 0.0
    for mask in _constant_masks(target):
        for channel in range(target.shape[2]):
            reference_values = reference[..., channel][mask]
            direct_values = direct[..., channel][mask]
            maximum_mean = max(
                maximum_mean,
                abs(
                    float(np.mean(reference_values))
                    - float(np.mean(direct_values))
                ),
            )
            reference_variance = float(np.var(reference_values))
            if reference_variance > 1e-10:
                ratio = float(np.var(direct_values)) / reference_variance
                minimum_variance = min(minimum_variance, ratio)
                maximum_variance = max(maximum_variance, ratio)
    return maximum_mean, minimum_variance, maximum_variance


def _minimum_map_correlation(
    reference: np.ndarray, direct: np.ndarray
) -> float:
    return min(
        float(
            np.corrcoef(
                reference[..., channel].ravel(),
                direct[..., channel].ravel(),
            )[0, 1]
        )
        for channel in range(reference.shape[2])
    )


def _contrast_ratio(
    target: np.ndarray, reference: np.ndarray, direct: np.ndarray
) -> float:
    values = np.unique(np.round(target[..., 0], decimals=8))
    low_mask = np.isclose(target[..., 0], values[0])
    high_mask = np.isclose(target[..., 0], values[-1])
    ratios = []
    for channel in range(target.shape[2]):
        reference_contrast = float(
            np.mean(reference[..., channel][high_mask])
            - np.mean(reference[..., channel][low_mask])
        )
        direct_contrast = float(
            np.mean(direct[..., channel][high_mask])
            - np.mean(direct[..., channel][low_mask])
        )
        ratios.append(direct_contrast / reference_contrast)
    return max(ratios, key=lambda value: abs(value - 1.0))


def _step_center(values: np.ndarray) -> float:
    profile = np.mean(values, axis=(0, 2))
    midpoint = 0.5 * (float(np.mean(profile[:4])) + float(np.mean(profile[-4:])))
    above = np.flatnonzero(profile >= midpoint)
    if above.size == 0:
        raise RuntimeError("step midpoint is not crossed")
    index = int(above[0])
    if index == 0:
        return 0.0
    left, right = float(profile[index - 1]), float(profile[index])
    if right == left:
        return float(index)
    return float(index - 1) + (midpoint - left) / (right - left)


def _weighted_centroid(
    weights: np.ndarray, center: tuple[float, float], radius: float
) -> tuple[float, float]:
    ys, xs = np.indices(weights.shape)
    roi = (
        (ys - center[0]) ** 2 + (xs - center[1]) ** 2
        <= (2.0 * radius) ** 2
    )
    positive = np.maximum(weights, 0.0) * roi
    total = float(np.sum(positive))
    if total <= 0.0:
        raise RuntimeError("island centroid has no positive mass")
    return (
        float(np.sum(ys * positive) / total),
        float(np.sum(xs * positive) / total),
    )


def _island_metrics(
    target: np.ndarray,
    reference: np.ndarray,
    direct: np.ndarray,
    *,
    radius: float,
) -> tuple[float, float]:
    height, width = target.shape[:2]
    centers = (
        (height / 2.0, width * 3.0 / 10.0, 1.0),
        (height / 2.0, width * 7.0 / 10.0, -1.0),
    )
    background = np.median(target, axis=(0, 1))
    ratios = []
    shifts = []
    for cy, cx, sign in centers:
        for channel in range(target.shape[2]):
            reference_weights = sign * (
                reference[..., channel] - background[channel]
            )
            direct_weights = sign * (
                direct[..., channel] - background[channel]
            )
            ys, xs = np.indices(target.shape[:2])
            roi = (
                (ys - cy) ** 2 + (xs - cx) ** 2 <= radius * radius
            )
            reference_excess = float(
                np.sum(np.maximum(reference_weights, 0.0)[roi])
            )
            direct_excess = float(
                np.sum(np.maximum(direct_weights, 0.0)[roi])
            )
            ratios.append(direct_excess / reference_excess)
            reference_centroid = _weighted_centroid(
                reference_weights, (cy, cx), radius
            )
            direct_centroid = _weighted_centroid(
                direct_weights, (cy, cx), radius
            )
            shifts.append(
                float(
                    np.hypot(
                        direct_centroid[0] - reference_centroid[0],
                        direct_centroid[1] - reference_centroid[1],
                    )
                )
            )
    return max(ratios, key=lambda value: abs(value - 1.0)), max(shifts)


def _evaluate_split(
    contract: dict[str, Any],
    parent: dict[str, Any],
    *,
    seed_offset: int,
) -> dict[str, Any]:
    scenes = _base_scenes(contract)
    base_profiles = tuple(
        replace(profile, boundary_mode="normalized-support-v1")
        for profile in profiles_from_contract(parent, seed_offset=seed_offset)
    )
    rows = []
    for factor in (int(value) for value in contract["lod_factors"]):
        profiles = compile_two_cumulant_profiles(
            base_profiles, pixel_size_factor=factor
        )
        for name, base_target in scenes.items():
            target = _area_mean(base_target, factor)
            reference_transmittance = _area_mean(
                render_density_conditioned_structure(
                    base_target, base_profiles
                ).transmittance,
                factor,
            )
            reference = -np.log(reference_transmittance)
            direct = render_density_conditioned_structure(target, profiles)
            repeat = render_density_conditioned_structure(target, profiles)
            assembled = np.empty_like(direct.density)
            for y0 in range(0, target.shape[0], 31):
                y1 = min(target.shape[0], y0 + 31)
                assembled[y0:y1] = (
                    render_density_conditioned_structure_region(
                        target,
                        profiles,
                        origin_yx=(y0, 0),
                        shape=(y1 - y0, target.shape[1]),
                    ).density
                )
            constant_target = np.broadcast_to(
                np.mean(target, axis=(0, 1), keepdims=True),
                target.shape,
            ).copy()
            constant = render_density_conditioned_structure(
                constant_target, profiles
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
                        seed=51971 + seed_offset + factor,
                    )
                ),
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
                    radius=float(contract["scenes"]["islands"]["base_radius"])
                    / factor,
                )
                row["island_excess_ratio"] = ratio
                row["island_centroid_shift_pixels"] = shift
            rows.append(row)
    return {"rows": rows}


def evaluate_two_cumulant_nonstationarity(
    contract: dict[str, Any], parent: dict[str, Any]
) -> dict[str, Any]:
    scene = contract["scenes"]
    development = _evaluate_split(
        contract,
        parent,
        seed_offset=int(scene["development_seed_offset"]),
    )
    confirmation = _evaluate_split(
        contract,
        parent,
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
    "evaluate_two_cumulant_nonstationarity",
    "load_contract",
]
