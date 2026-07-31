"""Frozen U6.P4AH covariance-moment LOD evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_derivative_conditioned_structure import profile_from_contract
from src.eval.sensitometry_primitive import build_operator
from src.film_physics.derivative_conditioned_structure import (
    DerivativeConditionedStructureProfile,
    gaussian_block_mean_variance_scale,
    render_derivative_conditioned_structure,
)


SCHEMA = "neuro_film.u6_p4ah_covariance_moment_lod_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4ah_covariance_moment_lod_report.v1"


class CovarianceMomentLodError(RuntimeError):
    """Raised when frozen P4AH inputs drift."""


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise CovarianceMomentLodError("P4AH parent path must be relative")
    return root / path


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    model, evaluation, gates = value.get("model", {}), value.get("evaluation", {}), value.get("gates", {})
    if (
        value.get("schema") != SCHEMA
        or model.get("fit_allowed")
        or model.get("post_result_retuning_allowed")
        or evaluation.get("shape") != [256, 384]
        or evaluation.get("lod_factors") != [2, 4]
        or evaluation.get("runs") != 2
        or gates.get("minimum_variance_ratio") != 0.7
        or gates.get("maximum_variance_ratio") != 1.3
        or gates.get("minimum_median_variance_error_improvement_over_direct") != 0.5
        or gates.get("minimum_worst_variance_error_improvement_over_direct") != 0.1
        or gates.get("minimum_median_variance_error_improvement_over_unscaled") != 0.1
        or not gates.get("no_parameter_fit")
    ):
        raise CovarianceMomentLodError("P4AH frozen contract drift")
    return value


def _parents(config: dict[str, Any], root: Path) -> tuple[dict, dict]:
    loaded = {}
    for key in ("p4ag_decision", "p4af_contract", "sensitometry_contract"):
        path = _relative(root, config["parents"][f"{key}_path"])
        if not path.is_file() or _hash(path) != config["parents"][f"{key}_sha256"]:
            raise CovarianceMomentLodError(f"P4AH parent mismatch: {key}")
        loaded[key] = json.loads(path.read_text(encoding="utf-8"))
    if loaded["p4ag_decision"].get("decision") != "retain_full_resolution_reference_close_direct_lod":
        raise CovarianceMomentLodError("P4AG decision does not authorize P4AH")
    return loaded["p4af_contract"], loaded["sensitometry_contract"]


def _block_mean(values: np.ndarray, factor: int) -> np.ndarray:
    height, width, channels = values.shape
    return values.reshape(height // factor, factor, width // factor, factor, channels).mean(axis=(1, 3), dtype=np.float64)


def _profile(
    parent: DerivativeConditionedStructureProfile,
    factor: int,
    variance_scale: float,
) -> DerivativeConditionedStructureProfile:
    return DerivativeConditionedStructureProfile(
        peak_density_variance=parent.peak_density_variance * variance_scale,
        correlation_sigma_pixels=parent.correlation_sigma_pixels / factor,
        layer_seeds=parent.layer_seeds,
        exposure_minimum=parent.exposure_minimum,
        exposure_maximum=parent.exposure_maximum,
        normalization_samples=parent.normalization_samples,
    )


def _lag1(values: np.ndarray) -> float:
    return float(np.corrcoef(values[:, :-1].ravel(), values[:, 1:].ravel())[0, 1])


def evaluate_covariance_moment_lod(config: dict[str, Any], root: Path) -> dict[str, Any]:
    p4af_contract, sensitometry = _parents(config, root)
    operator = build_operator(sensitometry)
    parent = profile_from_contract(p4af_contract)
    evaluation, gates = config["evaluation"], config["gates"]
    height, width = (int(item) for item in evaluation["shape"])
    rows, direct_improvements, unscaled_improvements = [], [], []
    maximum_mean = maximum_acf = 0.0
    minimum_ratio, maximum_ratio = float("inf"), 0.0
    repeat_exact = True
    physical_domain = True

    for factor in evaluation["lod_factors"]:
        factor = int(factor)
        analytic_scale = gaussian_block_mean_variance_scale(parent.correlation_sigma_pixels, factor)
        direct_scale = 1.0 / float(factor * factor)
        if not direct_scale < analytic_scale < 1.0:
            raise RuntimeError("P4AH analytic scale is outside frozen controls")
        for level in evaluation["flat_exposure_levels"]:
            exposure = np.full((height, width, 3), float(level), dtype=np.float64)
            truth = render_derivative_conditioned_structure(exposure, operator, parent)
            truth_density = _block_mean(truth.density.astype(np.float64), factor)
            low_exposure = _block_mean(exposure, factor)
            candidate_profile = _profile(parent, factor, analytic_scale)
            candidate = render_derivative_conditioned_structure(low_exposure, operator, candidate_profile)
            repeat = render_derivative_conditioned_structure(low_exposure, operator, candidate_profile)
            repeat_exact &= np.array_equal(candidate.density, repeat.density)
            repeat_exact &= np.array_equal(candidate.transmittance, repeat.transmittance)
            physical_domain &= bool(
                np.min(candidate.density) >= 0.0
                and np.min(candidate.transmittance) > 0.0
                and np.max(candidate.transmittance) <= 1.0
            )
            direct = render_derivative_conditioned_structure(low_exposure, operator, _profile(parent, factor, direct_scale))
            unscaled = render_derivative_conditioned_structure(low_exposure, operator, _profile(parent, factor, 1.0))
            channel_rows = []
            for channel in range(3):
                reference = truth_density[..., channel]
                variants = {
                    "candidate": candidate.density[..., channel].astype(np.float64),
                    "direct": direct.density[..., channel].astype(np.float64),
                    "unscaled": unscaled.density[..., channel].astype(np.float64),
                }
                reference_variance = float(np.var(reference))
                ratios = {name: float(np.var(values)) / reference_variance for name, values in variants.items()}
                errors = {name: abs(math.log(ratio)) for name, ratio in ratios.items()}
                direct_improvement = 1.0 - errors["candidate"] / errors["direct"]
                unscaled_improvement = 1.0 - errors["candidate"] / errors["unscaled"]
                mean_error = abs(float(np.mean(variants["candidate"]) - np.mean(reference)))
                acf_error = abs(_lag1(variants["candidate"]) - _lag1(reference))
                maximum_mean = max(maximum_mean, mean_error)
                maximum_acf = max(maximum_acf, acf_error)
                minimum_ratio, maximum_ratio = min(minimum_ratio, ratios["candidate"]), max(maximum_ratio, ratios["candidate"])
                direct_improvements.append(direct_improvement)
                unscaled_improvements.append(unscaled_improvement)
                channel_rows.append({
                    "channel": channel,
                    "mean_density_absolute_error": mean_error,
                    "lag1_acf_absolute_error": acf_error,
                    "variance_ratios": ratios,
                    "variance_error_improvement_over_direct": direct_improvement,
                    "variance_error_improvement_over_unscaled": unscaled_improvement,
                })
            rows.append({
                "factor": factor,
                "exposure": float(level),
                "analytic_variance_scale": analytic_scale,
                "direct_variance_scale": direct_scale,
                "channels": channel_rows,
            })

    metrics = {
        "rows": rows,
        "maximum_mean_density_absolute_error": maximum_mean,
        "minimum_variance_ratio": minimum_ratio,
        "maximum_variance_ratio": maximum_ratio,
        "maximum_lag1_acf_absolute_error": maximum_acf,
        "median_variance_error_improvement_over_direct": float(np.median(direct_improvements)),
        "worst_variance_error_improvement_over_direct": min(direct_improvements),
        "median_variance_error_improvement_over_unscaled": float(np.median(unscaled_improvements)),
        "density_transmittance_domain": physical_domain,
        "repeat_exact": repeat_exact,
    }
    checks = {
        "mean": maximum_mean <= gates["maximum_mean_density_absolute_error"],
        "variance": minimum_ratio >= gates["minimum_variance_ratio"] and maximum_ratio <= gates["maximum_variance_ratio"],
        "acf": maximum_acf <= gates["maximum_lag1_acf_absolute_error"],
        "beats_direct_median": metrics["median_variance_error_improvement_over_direct"] >= gates["minimum_median_variance_error_improvement_over_direct"],
        "beats_direct_worst": metrics["worst_variance_error_improvement_over_direct"] >= gates["minimum_worst_variance_error_improvement_over_direct"],
        "beats_unscaled_median": metrics["median_variance_error_improvement_over_unscaled"] >= gates["minimum_median_variance_error_improvement_over_unscaled"],
        "analytic_scale_order": all(row["direct_variance_scale"] < row["analytic_variance_scale"] < 1.0 for row in rows),
        "physical_domain": physical_domain,
        "repeat_exact": repeat_exact,
        "no_parameter_fit": True,
    }
    passed = all(checks.values())
    stable = {"experiment_id": config["experiment_id"], "metrics": metrics, "checks": checks, "passed": passed}
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
        "decision": config["branch_rule"]["pass" if passed else "fail"],
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = ["CovarianceMomentLodError", "REPORT_SCHEMA", "SCHEMA", "evaluate_covariance_moment_lod", "load_contract"]
