"""U6.P4CG Boolean-reference versus exact-spectrum Gaussian LOD diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import label

from src.filmfx.boolean_grain import (
    build_boolean_grain_context,
    render_boolean_grain,
)

SCHEMA = "neuro_film.u6_p4cg_boolean_gaussian_lod_boundary_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4cg_boolean_gaussian_lod_boundary_report.v1"


class BooleanGaussianLodError(RuntimeError):
    """Raised when the frozen P4CG contract or a numerical invariant drifts."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BooleanGaussianLodError(f"expected JSON object: {path}")
    return value


def validate_contract(contract: Mapping[str, Any], root: Path) -> None:
    reference = contract.get("reference", {})
    features = contract.get("higher_order_features", {})
    gates = contract.get("automatic_gates", {})
    if (
        contract.get("schema") != SCHEMA
        or reference.get("input_shape") != [24, 24]
        or reference.get("input_intensity") != 0.5
        or reference.get("output_zoom") != 4
        or reference.get("radius_input_pixels") != [0.22, 0.38, 0.7]
        or reference.get("monte_carlo_samples") != 32
        or reference.get("edge_crop_output_pixels") != 12
        or len(reference.get("development_seeds", [])) != 8
        or len(reference.get("confirmation_seeds", [])) != 8
        or len(reference.get("gaussian_confirmation_seeds", [])) != 8
        or features.get("local_rms_block_pixels") != 8
        or features.get("excursion_thresholds_sigma") != [2.0, 2.5]
        or gates.get("maximum_small_radius_gaussian_to_boolean_distance_ratio") != 1.5
        or gates.get("minimum_large_radius_gaussian_to_boolean_distance_ratio") != 2.0
        or gates.get("minimum_large_minus_small_distance_ratio") != 0.75
        or gates.get("minimum_large_radius_feature_families_diverging") != 2
        or gates.get("require_two_byte_identical_reports") is not True
    ):
        raise BooleanGaussianLodError("P4CG frozen contract drift")

    parents = contract["parents"]
    boolean = _load_json(root / parents["boolean_representation_decision"]["path"])
    visual = _load_json(root / parents["boolean_visual_decision"]["path"])
    higher = _load_json(root / parents["real_uniform_higher_order_decision"]["path"])
    if (
        boolean.get("decision")
        != parents["boolean_representation_decision"]["required_decision"]
        or visual.get("status") != parents["boolean_visual_decision"]["required_status"]
        or higher.get("decision")
        != parents["real_uniform_higher_order_decision"]["required_decision"]
    ):
        raise BooleanGaussianLodError("P4CG parent decision drift")


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _standardize(values: np.ndarray) -> np.ndarray:
    field = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(field, dtype=np.float64))
    centered = field - mean
    rms = float(np.sqrt(np.mean(np.square(centered), dtype=np.float64)))
    if not math.isfinite(rms) or rms <= 0.0:
        raise BooleanGaussianLodError("field variance is zero or non-finite")
    result = np.ascontiguousarray(centered / rms, dtype=np.float64)
    result.setflags(write=False)
    return result


def _render_boolean_interior(
    contract: Mapping[str, Any], *, radius: float, seed: int
) -> tuple[np.ndarray, int, str]:
    reference = contract["reference"]
    shape = tuple(int(value) for value in reference["input_shape"])
    intensity = np.full(shape, float(reference["input_intensity"]), dtype=np.float64)
    context = build_boolean_grain_context(
        intensity,
        radius_input_pixels=float(radius),
        monte_carlo_samples=int(reference["monte_carlo_samples"]),
        gaussian_filter_sigma_output_pixels=float(
            reference["gaussian_filter_sigma_output_pixels"]
        ),
        maximum_input_intensity=float(reference["maximum_input_intensity"]),
        epsilon=float(reference["epsilon"]),
        seed=int(seed),
    )
    rendered = render_boolean_grain(context, output_zoom=int(reference["output_zoom"]))
    edge = int(reference["edge_crop_output_pixels"])
    interior = rendered[edge:-edge, edge:-edge]
    return _standardize(interior), context.grain_count, context.fingerprint()


def _periodogram(values: np.ndarray) -> np.ndarray:
    field = np.asarray(values, dtype=np.float64)
    return np.square(np.abs(np.fft.fft2(field))) / field.size


def _development_spectrum(fields: Sequence[np.ndarray]) -> np.ndarray:
    if not fields:
        raise BooleanGaussianLodError("development spectrum has no fields")
    spectrum = np.mean(
        np.asarray([_periodogram(field) for field in fields], dtype=np.float64),
        axis=0,
        dtype=np.float64,
    )
    spectrum[0, 0] = 0.0
    mean = float(np.mean(spectrum, dtype=np.float64))
    if not math.isfinite(mean) or mean <= 0.0:
        raise BooleanGaussianLodError("development spectrum has no energy")
    spectrum = np.ascontiguousarray(spectrum / mean, dtype=np.float64)
    spectrum.setflags(write=False)
    return spectrum


def synthesize_exact_spectrum_gaussian(
    spectrum: np.ndarray, *, seed: int
) -> tuple[np.ndarray, float]:
    """Return a real Gaussian-phase field with the exact frozen periodogram."""
    target = np.asarray(spectrum, dtype=np.float64)
    if (
        target.ndim != 2
        or not np.all(np.isfinite(target))
        or np.any(target < 0.0)
        or target[0, 0] != 0.0
    ):
        raise ValueError("spectrum must be finite, nonnegative and zero-DC")
    white = np.random.default_rng(int(seed)).normal(size=target.shape)
    coefficients = np.fft.fft2(white)
    magnitude = np.abs(coefficients)
    phase = np.divide(
        coefficients,
        magnitude,
        out=np.zeros_like(coefficients),
        where=magnitude > 0.0,
    )
    coefficients = phase * np.sqrt(target * target.size)
    coefficients[0, 0] = 0.0
    complex_field = np.fft.ifft2(coefficients)
    field = np.ascontiguousarray(complex_field.real, dtype=np.float64)
    actual = _periodogram(field)
    relative_error = float(
        np.max(np.abs(actual - target))
        / max(float(np.max(target)), np.finfo(float).tiny)
    )
    field.setflags(write=False)
    return field, relative_error


def higher_order_features(
    values: np.ndarray, contract: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    field = _standardize(values)
    definition = contract["higher_order_features"]
    marginal = np.quantile(
        field,
        np.asarray(definition["marginal_quantile_probabilities"], dtype=np.float64),
        method="linear",
    )
    block = int(definition["local_rms_block_pixels"])
    if field.shape[0] % block or field.shape[1] % block:
        raise BooleanGaussianLodError("field is not divisible by local RMS block")
    local_rms = np.sqrt(
        np.mean(
            np.square(
                field.reshape(
                    field.shape[0] // block,
                    block,
                    field.shape[1] // block,
                    block,
                )
            ),
            axis=(1, 3),
            dtype=np.float64,
        )
    )
    energy = np.quantile(
        local_rms,
        np.asarray(definition["local_rms_quantile_probabilities"], dtype=np.float64),
        method="linear",
    )

    connectivity = np.ones((3, 3), dtype=np.uint8)
    per_megapixel = 1_000_000.0 / field.size
    topology: list[float] = []
    for threshold in definition["excursion_thresholds_sigma"]:
        for mask in (field >= float(threshold), field <= -float(threshold)):
            components, count = label(mask, structure=connectivity)
            sizes = np.bincount(components.reshape(-1), minlength=count + 1)[1:]
            for lower, upper in definition["excursion_component_area_bins_pixels"]:
                selected = int(np.count_nonzero((sizes >= lower) & (sizes <= upper)))
                topology.append(math.log1p(selected * per_megapixel))
    result = {
        "marginal_quantiles": np.ascontiguousarray(marginal, dtype=np.float64),
        "local_rms_quantiles": np.ascontiguousarray(energy, dtype=np.float64),
        "excursion_topology": np.ascontiguousarray(topology, dtype=np.float64),
    }
    if any(not np.all(np.isfinite(item)) for item in result.values()):
        raise BooleanGaussianLodError("higher-order feature is non-finite")
    return result


def _feature_distance_summary(
    development: Sequence[dict[str, np.ndarray]],
    boolean_confirmation: Sequence[dict[str, np.ndarray]],
    gaussian_confirmation: Sequence[dict[str, np.ndarray]],
) -> dict[str, Any]:
    families: dict[str, Any] = {}
    ratios: list[float] = []
    for name in development[0]:
        dev = np.asarray([row[name] for row in development], dtype=np.float64)
        center = np.median(dev, axis=0)
        spread_vector = np.quantile(dev, 0.75, axis=0) - np.quantile(dev, 0.25, axis=0)
        spread = float(np.sqrt(np.mean(np.square(spread_vector), dtype=np.float64)))
        if not math.isfinite(spread) or spread < 0.0:
            raise BooleanGaussianLodError(f"invalid development spread for {name}")
        zero_development_spread = spread == 0.0
        distance_scale = 1.0 if zero_development_spread else spread

        def distances(
            rows: Sequence[dict[str, np.ndarray]],
            *,
            family_name: str = name,
            family_center: np.ndarray = center,
            family_spread: float = distance_scale,
        ) -> np.ndarray:
            return np.asarray(
                [
                    math.sqrt(
                        float(
                            np.mean(
                                np.square(np.asarray(row[family_name]) - family_center),
                                dtype=np.float64,
                            )
                        )
                    )
                    / family_spread
                    for row in rows
                ],
                dtype=np.float64,
            )

        boolean_distance = distances(boolean_confirmation)
        gaussian_distance = distances(gaussian_confirmation)
        boolean_median = float(np.median(boolean_distance))
        gaussian_median = float(np.median(gaussian_distance))
        zero_boolean_distance = boolean_median == 0.0
        if zero_boolean_distance:
            ratio = 0.0 if gaussian_median == 0.0 else 1e300
        else:
            ratio = gaussian_median / boolean_median
        ratios.append(ratio)
        families[name] = {
            "development_iqr_rms_scale": spread,
            "development_iqr_rms_scale_was_zero": zero_development_spread,
            "distance_scale_used": distance_scale,
            "boolean_confirmation_median_distance": boolean_median,
            "gaussian_confirmation_median_distance": gaussian_median,
            "gaussian_to_boolean_distance_ratio": ratio,
            "boolean_confirmation_median_distance_was_zero": zero_boolean_distance,
        }
    return {
        "families": families,
        "median_gaussian_to_boolean_distance_ratio": float(np.median(ratios)),
        "families_with_ratio_at_least_two": int(
            np.count_nonzero(np.asarray(ratios) >= 2.0)
        ),
    }


def evaluate_boolean_gaussian_lod_boundary(
    contract: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    validate_contract(contract, root)
    reference = contract["reference"]
    radius_rows: list[dict[str, Any]] = []
    maximum_spectrum_error = 0.0
    for radius in reference["radius_input_pixels"]:
        development_fields: list[np.ndarray] = []
        development_features: list[dict[str, np.ndarray]] = []
        development_hashes: list[str] = []
        grain_counts: list[int] = []
        context_ids: list[str] = []
        for seed in reference["development_seeds"]:
            field, count, context_id = _render_boolean_interior(
                contract, radius=float(radius), seed=int(seed)
            )
            development_fields.append(field)
            development_features.append(higher_order_features(field, contract))
            development_hashes.append(_sha256_array(field))
            grain_counts.append(count)
            context_ids.append(context_id)
        spectrum = _development_spectrum(development_fields)

        boolean_features: list[dict[str, np.ndarray]] = []
        boolean_hashes: list[str] = []
        for seed in reference["confirmation_seeds"]:
            field, count, context_id = _render_boolean_interior(
                contract, radius=float(radius), seed=int(seed)
            )
            boolean_features.append(higher_order_features(field, contract))
            boolean_hashes.append(_sha256_array(field))
            grain_counts.append(count)
            context_ids.append(context_id)

        gaussian_features: list[dict[str, np.ndarray]] = []
        gaussian_hashes: list[str] = []
        spectrum_errors: list[float] = []
        for seed in reference["gaussian_confirmation_seeds"]:
            field, error = synthesize_exact_spectrum_gaussian(spectrum, seed=int(seed))
            gaussian_features.append(higher_order_features(field, contract))
            gaussian_hashes.append(_sha256_array(field))
            spectrum_errors.append(error)
        radius_error = max(spectrum_errors)
        maximum_spectrum_error = max(maximum_spectrum_error, radius_error)
        comparison = _feature_distance_summary(
            development_features, boolean_features, gaussian_features
        )
        radius_rows.append(
            {
                "radius_input_pixels": float(radius),
                "radius_output_pixels": float(radius) * int(reference["output_zoom"]),
                "development_spectrum_sha256": _sha256_array(spectrum),
                "maximum_gaussian_periodogram_relative_error": radius_error,
                "minimum_grain_count": min(grain_counts),
                "maximum_grain_count": max(grain_counts),
                "all_context_ids_distinct": len(set(context_ids)) == len(context_ids),
                "all_field_hashes_distinct": len(
                    set(development_hashes + boolean_hashes + gaussian_hashes)
                )
                == len(development_hashes + boolean_hashes + gaussian_hashes),
                "comparison": comparison,
            }
        )

    small = radius_rows[0]["comparison"]
    large = radius_rows[-1]["comparison"]
    small_ratio = float(small["median_gaussian_to_boolean_distance_ratio"])
    large_ratio = float(large["median_gaussian_to_boolean_distance_ratio"])
    gates = contract["automatic_gates"]
    checks = {
        "small_radius_gaussian_adequacy": small_ratio
        <= float(gates["maximum_small_radius_gaussian_to_boolean_distance_ratio"]),
        "large_radius_gaussian_divergence": large_ratio
        >= float(gates["minimum_large_radius_gaussian_to_boolean_distance_ratio"]),
        "scale_separation": large_ratio - small_ratio
        >= float(gates["minimum_large_minus_small_distance_ratio"]),
        "large_radius_multiple_feature_divergence": int(
            large["families_with_ratio_at_least_two"]
        )
        >= int(gates["minimum_large_radius_feature_families_diverging"]),
        "exact_second_order_control": maximum_spectrum_error
        <= float(gates["maximum_second_order_periodogram_relative_error"]),
        "finite_nonzero_and_distinct": all(
            row["minimum_grain_count"] > 0
            and row["all_context_ids_distinct"]
            and row["all_field_hashes_distinct"]
            for row in radius_rows
        ),
    }
    automatic_pass = bool(all(checks.values()))
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": hashlib.sha256(canonical_json(contract)).hexdigest(),
        "radius_results": radius_rows,
        "maximum_gaussian_periodogram_relative_error": maximum_spectrum_error,
        "small_radius_distance_ratio": small_ratio,
        "large_radius_distance_ratio": large_ratio,
        "large_minus_small_distance_ratio": large_ratio - small_ratio,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_boolean_to_gaussian_scale_lod_distinction"
            if automatic_pass
            else "close_boolean_to_gaussian_scale_boundary_without_rescue"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable = dict(report)
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(stable)).hexdigest()
    return report


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "BooleanGaussianLodError",
    "evaluate_boolean_gaussian_lod_boundary",
    "higher_order_features",
    "synthesize_exact_spectrum_gaussian",
    "validate_contract",
    "write_report",
]
