"""Frozen U6.P4AF derivative-conditioned density-structure evaluator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from src.eval.physical_callier_source import hash_file
from src.eval.sensitometry_primitive import build_operator
from src.film_physics.derivative_conditioned_structure import (
    DerivativeConditionedStructureProfile,
    derivative_variance_shape,
    render_derivative_conditioned_structure,
    render_derivative_conditioned_structure_region,
)


SCHEMA = "neuro_film.u6_p4af_derivative_conditioned_structure_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4af_derivative_conditioned_structure_report.v1"


class DerivativeConditionedStructureError(RuntimeError):
    """Raised when P4AF contract or frozen inputs drift."""


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload.get("model", {})
    evaluation = payload.get("evaluation", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or model.get("marginal_family") != "moment-matched gamma density"
        or model.get("peak_density_variance") != 0.0001
        or model.get("correlation_sigma_pixels") != 0.65
        or model.get("layer_seeds") != [260831, 260837, 260851]
        or model.get("variance_normalization_domain") != [0.0001, 16.0]
        or model.get("variance_normalization_samples") != 32769
        or model.get("display_rgb_noise_allowed")
        or model.get("per_image_normalization_allowed")
        or model.get("hard_clipping_allowed")
        or evaluation.get("shape") != [512, 768]
        or evaluation.get("flat_exposure_levels")
        != [0.0, 0.001, 0.0055, 0.18, 4.0]
        or evaluation.get("ramp_bins") != 24
        or evaluation.get("row_partitions") != [31, 127]
        or evaluation.get("runs") != 2
        or evaluation.get("post_result_retuning_allowed")
        or gates.get("maximum_flat_mean_absolute_error") != 0.003
        or gates.get("maximum_flat_variance_relative_error") != 0.2
        or gates.get("minimum_ramp_variance_shape_spearman") != 0.9
        or gates.get("maximum_ramp_variance_relative_error_p90") != 0.3
        or gates.get("minimum_lag1_autocorrelation") != 0.05
        or not gates.get("no_parameter_fit")
    ):
        raise DerivativeConditionedStructureError("P4AF frozen contract drift")
    return payload


def _relative(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise DerivativeConditionedStructureError("P4AF parent path must be relative")
    return root / path


def _load_parents(
    config: dict[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    parents = config["parents"]
    identities: dict[str, str] = {}
    for stem in ("p4ae_decision", "sensitometry_contract"):
        path = _relative(root, parents[f"{stem}_path"])
        if not path.is_file():
            raise DerivativeConditionedStructureError(f"missing P4AF parent: {stem}")
        actual = hash_file(path)
        if actual != parents[f"{stem}_sha256"]:
            raise DerivativeConditionedStructureError(
                f"P4AF parent hash mismatch: {stem}"
            )
        identities[stem] = actual
    decision = json.loads(
        _relative(root, parents["p4ae_decision_path"]).read_text(encoding="utf-8")
    )
    if (
        decision.get("decision")
        != "open_synthetic_derivative_conditioned_structure_compiler"
        or not decision.get("passed")
    ):
        raise DerivativeConditionedStructureError("P4AE decision identity mismatch")
    sensitometry = json.loads(
        _relative(root, parents["sensitometry_contract_path"]).read_text(
            encoding="utf-8"
        )
    )
    return sensitometry, identities


def profile_from_contract(config: dict[str, Any]) -> DerivativeConditionedStructureProfile:
    model = config["model"]
    return DerivativeConditionedStructureProfile(
        peak_density_variance=float(model["peak_density_variance"]),
        correlation_sigma_pixels=float(model["correlation_sigma_pixels"]),
        layer_seeds=tuple(int(value) for value in model["layer_seeds"]),
        exposure_minimum=float(model["variance_normalization_domain"][0]),
        exposure_maximum=float(model["variance_normalization_domain"][1]),
        normalization_samples=int(model["variance_normalization_samples"]),
    )


def _lag1_correlation(values: np.ndarray) -> float:
    left = values[:, :-1].ravel()
    right = values[:, 1:].ravel()
    return float(np.corrcoef(left, right)[0, 1])


def evaluate_structure(config: dict[str, Any], root: Path) -> dict[str, Any]:
    sensitometry, identities = _load_parents(config, root)
    operator = build_operator(sensitometry)
    profile = profile_from_contract(config)
    evaluation = config["evaluation"]
    height, width = (int(value) for value in evaluation["shape"])
    border = int(evaluation["interior_border_pixels"])
    flat_reports = []
    maximum_mean_error = 0.0
    maximum_variance_relative_error = 0.0
    minimum_lag1 = float("inf")
    zero_exact = True

    for level in evaluation["flat_exposure_levels"]:
        source = np.full((height, width, 3), float(level), dtype=np.float64)
        preserved = source.copy()
        target, normalized_shape = derivative_variance_shape(source, operator, profile)
        rendered = render_derivative_conditioned_structure(source, operator, profile)
        interior = rendered.density[border:-border, border:-border].astype(np.float64)
        target_interior = target[border:-border, border:-border]
        residual = interior - target_interior
        expected_variance = (
            profile.peak_density_variance
            * normalized_shape[border:-border, border:-border]
        )
        mean_error = np.abs(np.mean(interior, axis=(0, 1)) - target[0, 0])
        empirical_variance = np.mean(np.square(residual), axis=(0, 1))
        active = expected_variance[0, 0] > 0.0
        variance_error = np.zeros(3, dtype=np.float64)
        variance_error[active] = np.abs(
            empirical_variance[active] - expected_variance[0, 0, active]
        ) / expected_variance[0, 0, active]
        lag1 = [
            _lag1_correlation(residual[..., channel])
            if active[channel]
            else 1.0
            for channel in range(3)
        ]
        maximum_mean_error = max(maximum_mean_error, float(np.max(mean_error)))
        if np.any(active):
            maximum_variance_relative_error = max(
                maximum_variance_relative_error,
                float(np.max(variance_error[active])),
            )
            minimum_lag1 = min(
                minimum_lag1,
                min(lag1[channel] for channel in range(3) if active[channel]),
            )
        if float(level) == 0.0:
            zero_exact = bool(np.array_equal(rendered.density, target.astype(np.float32)))
        if not np.array_equal(source, preserved):
            raise RuntimeError("P4AF renderer modified its input")
        flat_reports.append(
            {
                "exposure": float(level),
                "target_density": target[0, 0].tolist(),
                "normalized_variance_shape": normalized_shape[0, 0].tolist(),
                "mean_absolute_error": mean_error.tolist(),
                "empirical_variance": empirical_variance.tolist(),
                "expected_variance": expected_variance[0, 0].tolist(),
                "variance_relative_error": variance_error.tolist(),
                "lag1_autocorrelation": lag1,
            }
        )

    ramp_values = np.geomspace(
        float(evaluation["ramp_exposure_interval"][0]),
        float(evaluation["ramp_exposure_interval"][1]),
        width,
        dtype=np.float64,
    )
    ramp = np.broadcast_to(ramp_values[None, :, None], (height, width, 3)).copy()
    preserved_ramp = ramp.copy()
    target, normalized_shape = derivative_variance_shape(ramp, operator, profile)
    first = render_derivative_conditioned_structure(ramp, operator, profile)
    second = render_derivative_conditioned_structure(ramp, operator, profile)
    repeat_exact = bool(
        np.array_equal(first.density, second.density)
        and np.array_equal(first.transmittance, second.transmittance)
    )
    assembled_density = np.empty_like(first.density)
    assembled_transmittance = np.empty_like(first.transmittance)
    partition_exact = True
    for rows in evaluation["row_partitions"]:
        for y0 in range(0, height, int(rows)):
            y1 = min(height, y0 + int(rows))
            region = render_derivative_conditioned_structure_region(
                ramp,
                operator,
                profile,
                origin_yx=(y0, 0),
                shape=(y1 - y0, width),
            )
            assembled_density[y0:y1] = region.density
            assembled_transmittance[y0:y1] = region.transmittance
        partition_exact &= bool(
            np.array_equal(assembled_density, first.density)
            and np.array_equal(assembled_transmittance, first.transmittance)
        )

    ramp_observed = []
    ramp_expected = []
    bin_edges = np.linspace(0, width, int(evaluation["ramp_bins"]) + 1, dtype=int)
    residual = first.density.astype(np.float64) - target
    for channel in range(3):
        for index in range(len(bin_edges) - 1):
            x0, x1 = int(bin_edges[index]), int(bin_edges[index + 1])
            observed = float(
                np.mean(np.square(residual[border:-border, x0:x1, channel]))
            )
            expected = float(
                profile.peak_density_variance
                * np.mean(normalized_shape[border:-border, x0:x1, channel])
            )
            ramp_observed.append(observed)
            ramp_expected.append(expected)
    ramp_observed_array = np.asarray(ramp_observed, dtype=np.float64)
    ramp_expected_array = np.asarray(ramp_expected, dtype=np.float64)
    ramp_relative_error = np.abs(ramp_observed_array - ramp_expected_array) / (
        ramp_expected_array
    )
    ramp_spearman = float(
        spearmanr(ramp_expected_array, ramp_observed_array).statistic
    )
    ramp_p90 = float(np.quantile(ramp_relative_error, 0.9))

    metrics = {
        "flat_reports": flat_reports,
        "maximum_flat_mean_absolute_error": maximum_mean_error,
        "maximum_flat_variance_relative_error": maximum_variance_relative_error,
        "minimum_lag1_autocorrelation": minimum_lag1,
        "ramp_variance_shape_spearman": ramp_spearman,
        "ramp_variance_relative_error_p90": ramp_p90,
        "zero_exposure_density_exact": zero_exact,
        "density_minimum": float(np.min(first.density)),
        "transmittance_minimum": float(np.min(first.transmittance)),
        "transmittance_maximum": float(np.max(first.transmittance)),
        "input_unchanged": bool(np.array_equal(ramp, preserved_ramp)),
        "repeat_exact": repeat_exact,
        "row_partition_exact": partition_exact,
    }
    gates = config["gates"]
    gate_results = {
        "flat_mean": maximum_mean_error
        <= gates["maximum_flat_mean_absolute_error"],
        "flat_variance": maximum_variance_relative_error
        <= gates["maximum_flat_variance_relative_error"],
        "ramp_shape": ramp_spearman
        >= gates["minimum_ramp_variance_shape_spearman"],
        "ramp_variance": ramp_p90
        <= gates["maximum_ramp_variance_relative_error_p90"],
        "spatial_correlation": minimum_lag1
        >= gates["minimum_lag1_autocorrelation"],
        "zero_exposure": zero_exact,
        "density_domain": metrics["density_minimum"] >= 0.0,
        "transmittance_domain": bool(
            metrics["transmittance_minimum"] > 0.0
            and metrics["transmittance_maximum"] <= 1.0
        ),
        "input_unchanged": metrics["input_unchanged"],
        "repeat_exact": repeat_exact,
        "row_partition_exact": partition_exact,
        "no_parameter_fit": True,
        "parent_identity": True,
    }
    passed = all(gate_results.values())
    stable = {
        "experiment_id": config["experiment_id"],
        "parent_sha256": identities,
        "profile": {
            "peak_density_variance": profile.peak_density_variance,
            "correlation_sigma_pixels": profile.correlation_sigma_pixels,
            "layer_seeds": list(profile.layer_seeds),
        },
        "metrics": metrics,
        "gate_results": gate_results,
        "passed": passed,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            "retain_generic_derivative_conditioned_structure"
            if passed
            else "close_derivative_conditioned_structure_without_rescue"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "DerivativeConditionedStructureError",
    "evaluate_structure",
    "load_contract",
    "profile_from_contract",
]
