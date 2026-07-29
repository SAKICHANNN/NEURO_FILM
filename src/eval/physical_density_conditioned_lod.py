"""U6.P4E physical-LOD audit for density-conditioned material structure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.eval.physical_density_conditioned_structure import (
    profiles_from_contract,
)
from src.film_physics.density_conditioned_structure import (
    compile_density_conditioned_profiles,
    render_density_conditioned_structure,
    render_density_conditioned_structure_region,
)
from src.film_physics.structure_compiler import counter_normal_region


SCHEMA = "neuro_film.u6_p4e_density_conditioned_lod_audit_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4e_density_conditioned_lod_audit_report.v1"
ProfileCompiler = Callable[..., tuple[Any, ...]]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_exact(root: Path, path: str, expected: str) -> dict[str, Any]:
    absolute = root / path
    if _sha256(absolute) != expected:
        raise ValueError(f"U6.P4E parent drift: {path}")
    return json.loads(absolute.read_text(encoding="utf-8"))


def load_contract(
    root: Path, path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _sha256(path) != expected_sha256:
        raise ValueError("U6.P4E contract hash mismatch")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != SCHEMA
        or contract["training_allowed"]
        or contract["photograph_access_allowed"]
        or contract["production_integration_allowed"]
        or contract["compiler"]["per_image_fit_or_normalization_allowed"]
        or contract["controls"]["display_rgb_clipping_allowed"]
    ):
        raise ValueError("unsupported U6.P4E contract")
    parents = contract["parents"]
    parent = _load_exact(
        root,
        parents["p4d_contract"],
        parents["p4d_contract_sha256"],
    )
    decision = _load_exact(
        root,
        parents["p4d_decision"],
        parents["p4d_decision_sha256"],
    )
    if (
        not parent["branch_rule"]["pass"].startswith("retain")
        or decision["decision"]
        != "retain_generic_density_conditioned_material_structure_open_integration_lod_audit"
    ):
        raise ValueError("U6.P4E parent did not open this leaf")
    return contract, parent


def _scene(contract: dict[str, Any]) -> np.ndarray:
    spec = contract["synthetic_scene"]
    maximum_factor = max(int(value) for value in contract["lod_factors"])
    coarse_height, coarse_width = (
        int(value) for value in spec["coarse_shape_at_factor_4"]
    )
    height = coarse_height * maximum_factor
    width = coarse_width * maximum_factor
    horizontal = np.linspace(
        0.0,
        float(spec["maximum_target_density"]),
        width,
        dtype=np.float64,
    )[None, :]
    low, high = (float(value) for value in spec["vertical_density_modulation"])
    vertical = np.linspace(low, high, height, dtype=np.float64)[:, None]
    base = horizontal * vertical
    scales = np.asarray(spec["channel_density_scales"], dtype=np.float64)
    target = np.clip(
        base[..., None] * scales[None, None, :],
        0.0,
        float(spec["maximum_target_density"]),
    )
    return target


def _area_mean(values: np.ndarray, factor: int) -> np.ndarray:
    height, width, channels = values.shape
    if height % factor or width % factor:
        raise ValueError("LOD reference dimensions are not factor divisible")
    return values.reshape(
        height // factor,
        factor,
        width // factor,
        factor,
        channels,
    ).mean(axis=(1, 3), dtype=np.float64)


def _bin_metrics(
    target: np.ndarray,
    reference: np.ndarray,
    direct: np.ndarray,
    bins: list[float],
) -> dict[str, float]:
    maximum_mean_difference = 0.0
    maximum_target_error = 0.0
    minimum_variance_ratio = float("inf")
    maximum_variance_ratio = 0.0
    for channel in range(target.shape[2]):
        for lower, upper in zip(bins[:-1], bins[1:], strict=True):
            mask = (target[..., channel] >= lower) & (
                target[..., channel] < upper
            )
            if np.count_nonzero(mask) < 32:
                continue
            target_mean = float(np.mean(target[..., channel][mask]))
            reference_values = reference[..., channel][mask]
            direct_values = direct[..., channel][mask]
            reference_mean = float(np.mean(reference_values))
            direct_mean = float(np.mean(direct_values))
            maximum_mean_difference = max(
                maximum_mean_difference, abs(direct_mean - reference_mean)
            )
            maximum_target_error = max(
                maximum_target_error, abs(direct_mean - target_mean)
            )
            reference_variance = float(np.var(reference_values))
            direct_variance = float(np.var(direct_values))
            if reference_variance > 1e-10:
                ratio = direct_variance / reference_variance
                minimum_variance_ratio = min(minimum_variance_ratio, ratio)
                maximum_variance_ratio = max(maximum_variance_ratio, ratio)
    return {
        "maximum_direct_to_reference_local_mean_difference": (
            maximum_mean_difference
        ),
        "maximum_direct_local_mean_target_error": maximum_target_error,
        "minimum_direct_to_reference_variance_ratio": minimum_variance_ratio,
        "maximum_direct_to_reference_variance_ratio": maximum_variance_ratio,
    }


def _column_correlation(target: np.ndarray, values: np.ndarray) -> float:
    correlations = []
    for channel in range(target.shape[2]):
        target_columns = np.mean(target[..., channel], axis=0)
        value_columns = np.mean(values[..., channel], axis=0)
        correlations.append(
            float(np.corrcoef(target_columns, value_columns)[0, 1])
        )
    return min(correlations)


def _display_noise_out_of_domain(
    transmittance: np.ndarray, *, seed: int
) -> float:
    standard_deviation = float(np.std(transmittance, dtype=np.float64))
    channels = []
    for channel in range(transmittance.shape[2]):
        channels.append(
            counter_normal_region(
                transmittance.shape[:2],
                origin_yx=(0, 0),
                shape=transmittance.shape[:2],
                seed=seed + channel * 101,
            )
        )
    noise = standard_deviation * np.stack(channels, axis=-1)
    invalid = (transmittance + noise < 0.0) | (
        transmittance + noise > 1.0
    )
    return float(np.mean(invalid))


def _evaluate_split(
    contract: dict[str, Any],
    parent: dict[str, Any],
    *,
    seed_offset: int,
    profile_compiler: ProfileCompiler,
) -> dict[str, Any]:
    target_base = _scene(contract)
    base_profiles = profiles_from_contract(parent, seed_offset=seed_offset)
    factors: list[dict[str, Any]] = []
    for factor in (int(value) for value in contract["lod_factors"]):
        target = _area_mean(target_base, factor)
        expanded_target = np.repeat(
            np.repeat(target, factor, axis=0), factor, axis=1
        )
        base_reference = render_density_conditioned_structure(
            expanded_target, base_profiles
        )
        reference_transmittance = _area_mean(
            base_reference.transmittance, factor
        )
        reference_density = -np.log(reference_transmittance)
        compiled_profiles = profile_compiler(
            base_profiles, pixel_size_factor=factor
        )
        direct = render_density_conditioned_structure(
            target, compiled_profiles
        )
        repeat = render_density_conditioned_structure(
            target, compiled_profiles
        )
        repeat_exact = bool(
            np.array_equal(direct.density, repeat.density)
            and np.array_equal(direct.transmittance, repeat.transmittance)
        )
        assembled_density = np.empty_like(direct.density)
        assembled_transmittance = np.empty_like(direct.transmittance)
        for y0 in range(0, target.shape[0], 31):
            y1 = min(target.shape[0], y0 + 31)
            region = render_density_conditioned_structure_region(
                target,
                compiled_profiles,
                origin_yx=(y0, 0),
                shape=(y1 - y0, target.shape[1]),
            )
            assembled_density[y0:y1] = region.density
            assembled_transmittance[y0:y1] = region.transmittance
        partition_exact = bool(
            np.array_equal(direct.density, assembled_density)
            and np.array_equal(
                direct.transmittance, assembled_transmittance
            )
        )
        constant_target = np.broadcast_to(
            np.mean(target, axis=(0, 1), keepdims=True), target.shape
        ).copy()
        constant = render_density_conditioned_structure(
            constant_target, compiled_profiles
        )
        bins = [float(value) for value in contract["synthetic_scene"]["density_bins"]]
        factors.append(
            {
                "factor": factor,
                **_bin_metrics(
                    target,
                    reference_density,
                    direct.density.astype(np.float64),
                    bins,
                ),
                "candidate_target_column_correlation": _column_correlation(
                    target, direct.density
                ),
                "constant_rate_target_column_correlation": (
                    _column_correlation(target, constant.density)
                ),
                "display_noise_out_of_domain_fraction": (
                    _display_noise_out_of_domain(
                        direct.transmittance, seed=99173 + seed_offset + factor
                    )
                ),
                "minimum_density": float(np.min(direct.density)),
                "maximum_transmittance": float(
                    np.max(direct.transmittance)
                ),
                "repeat_exact": repeat_exact,
                "row_partition_exact": partition_exact,
                "density_sha256": hashlib.sha256(
                    direct.density.tobytes(order="C")
                ).hexdigest(),
                "transmittance_sha256": hashlib.sha256(
                    direct.transmittance.tobytes(order="C")
                ).hexdigest(),
            }
        )
    return {"factors": factors}


def evaluate_density_conditioned_lod(
    contract: dict[str, Any], parent: dict[str, Any]
) -> dict[str, Any]:
    return evaluate_density_conditioned_lod_with_compiler(
        contract,
        parent,
        profile_compiler=compile_density_conditioned_profiles,
        report_schema=REPORT_SCHEMA,
    )


def evaluate_density_conditioned_lod_with_compiler(
    contract: dict[str, Any],
    parent: dict[str, Any],
    *,
    profile_compiler: ProfileCompiler,
    report_schema: str,
) -> dict[str, Any]:
    scene = contract["synthetic_scene"]
    development = _evaluate_split(
        contract,
        parent,
        seed_offset=int(scene["development_seed_offset"]),
        profile_compiler=profile_compiler,
    )
    confirmation = _evaluate_split(
        contract,
        parent,
        seed_offset=int(scene["confirmation_seed_offset"]),
        profile_compiler=profile_compiler,
    )
    rows = development["factors"] + confirmation["factors"]
    gates = contract["automatic_gates"]
    checks = {
        "candidate_column_correlation": min(
            row["candidate_target_column_correlation"] for row in rows
        )
        >= float(gates["minimum_candidate_target_column_correlation"]),
        "candidate_local_mean": max(
            row["maximum_direct_local_mean_target_error"] for row in rows
        )
        <= float(gates["maximum_candidate_local_mean_absolute_error"]),
        "reference_local_mean": max(
            row["maximum_direct_to_reference_local_mean_difference"]
            for row in rows
        )
        <= float(
            gates["maximum_direct_to_area_reference_local_mean_difference"]
        ),
        "reference_variance": min(
            row["minimum_direct_to_reference_variance_ratio"] for row in rows
        )
        >= float(
            gates["minimum_direct_to_area_reference_variance_ratio"]
        )
        and max(
            row["maximum_direct_to_reference_variance_ratio"] for row in rows
        )
        <= float(
            gates["maximum_direct_to_area_reference_variance_ratio"]
        ),
        "constant_rate_negative": max(
            abs(row["constant_rate_target_column_correlation"])
            for row in rows
        )
        <= float(gates["maximum_constant_rate_target_column_correlation"]),
        "display_noise_negative": min(
            row["display_noise_out_of_domain_fraction"] for row in rows
        )
        >= float(gates["minimum_display_noise_out_of_domain_fraction"]),
        "physical_domain": min(row["minimum_density"] for row in rows) >= 0.0
        and max(row["maximum_transmittance"] for row in rows) <= 1.0,
        "repeat_exact": all(row["repeat_exact"] for row in rows),
        "row_partition_exact": all(
            row["row_partition_exact"] for row in rows
        ),
    }
    return {
        "schema": report_schema,
        "node": contract["node"],
        "development": development,
        "confirmation": confirmation,
        "checks": checks,
        "automatic_pass": all(checks.values()),
        "decision": (
            contract["branch_rule"]["pass"]
            if all(checks.values())
            else contract["branch_rule"]["fail"]
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "evaluate_density_conditioned_lod",
    "evaluate_density_conditioned_lod_with_compiler",
    "load_contract",
]
