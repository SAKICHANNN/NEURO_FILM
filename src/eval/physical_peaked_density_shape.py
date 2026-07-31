"""No-fit P4AC audit of one source-stated peaked grain-variance shape."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from src.eval.physical_callier_source import hash_file


SCHEMA = "neuro_film.u6_p4ac_peaked_density_grain_shape_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4ac_peaked_density_grain_shape_report.v1"


class PeakedDensityShapeError(RuntimeError):
    """Raised when the P4AC contract or its frozen parents drift."""


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
        or model.get("peak_density") != 2.0
        or model.get("amplitude_fit_allowed")
        or model.get("table_value_fit_allowed")
        or evaluation.get("tables") != [4, 5, 6]
        or evaluation.get("apertures_um") != [13, 27, 57]
        or evaluation.get("minimum_points_per_series") != 4
        or evaluation.get("runs") != 2
        or evaluation.get("post_result_retuning_allowed")
        or gates.get("required_eligible_series") != 9
        or gates.get("candidate_median_normalized_relative_error_maximum")
        != 0.15
        or gates.get("candidate_p90_normalized_relative_error_maximum") != 0.3
        or gates.get("candidate_median_spearman_minimum") != 0.7
        or gates.get("minimum_relative_improvement_in_median_error_over_p4d")
        != 0.1
        or gates.get("maximum_candidate_to_p4d_p90_error_ratio") != 1.0
        or gates.get("observed_peak_distance_from_2d_maximum") != 0.75
        or gates.get("minimum_series_with_peak_near_2d") != 7
        or not gates.get("no_parameter_fit")
    ):
        raise PeakedDensityShapeError("P4AC frozen contract drift")
    return payload


def _relative(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise PeakedDensityShapeError("P4AC parent path must be relative")
    return root / path


def _load_parents(
    config: dict[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    parents = config["parents"]
    identities: dict[str, str] = {}
    for stem in ("p4ab_decision", "p4aa_contract", "p4aa_report"):
        path = _relative(root, parents[f"{stem}_path"])
        if not path.is_file():
            raise PeakedDensityShapeError(f"missing P4AC parent: {stem}")
        actual = hash_file(path)
        if actual != parents[f"{stem}_sha256"]:
            raise PeakedDensityShapeError(f"P4AC parent hash mismatch: {stem}")
        identities[stem] = actual
    decision = json.loads(
        _relative(root, parents["p4ab_decision_path"]).read_text(encoding="utf-8")
    )
    if (
        decision.get("schema")
        != "neuro_film.u6_p4ab_fixed_poisson_density_shape_decision.v1"
        or decision.get("decision")
        != "close_p4d_measured_shape_compatibility_without_retuning"
        or decision.get("passed") is not False
    ):
        raise PeakedDensityShapeError("P4AB decision identity mismatch")
    source = json.loads(
        _relative(root, parents["p4aa_report_path"]).read_text(encoding="utf-8")
    )
    if (
        source.get("schema")
        != "neuro_film.u6_p4aa_nasa_density_grain_source_report.v1"
        or source.get("stable_evidence_id") != parents["p4aa_stable_evidence_id"]
        or not source.get("source_pass")
    ):
        raise PeakedDensityShapeError("P4AA evidence identity mismatch")
    return source, identities


def _relative_error(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.abs(observed - predicted) / predicted


def evaluate_shape(config: dict[str, Any], root: Path) -> dict[str, Any]:
    source, identities = _load_parents(config, root)
    rows = source["numeric_density_granularity_rows"]
    peak_density = float(config["model"]["peak_density"])
    minimum_points = int(config["evaluation"]["minimum_points_per_series"])
    series: list[dict[str, Any]] = []
    candidate_errors: list[float] = []
    baseline_errors: list[float] = []
    candidate_correlations: list[float] = []
    peak_near_count = 0

    for table in config["evaluation"]["tables"]:
        table_rows = [row for row in rows if row["table"] == table]
        for aperture in config["evaluation"]["apertures_um"]:
            points = sorted(
                (
                    float(row["density"]),
                    float(row["rms_granularity_by_aperture_um"][str(aperture)]),
                )
                for row in table_rows
                if str(aperture) in row["rms_granularity_by_aperture_um"]
            )
            if len(points) < minimum_points:
                continue
            density = np.asarray([point[0] for point in points], dtype=np.float64)
            observed = np.asarray([point[1] for point in points], dtype=np.float64)
            if (
                not np.all(np.isfinite(density))
                or not np.all(np.isfinite(observed))
                or np.any(density <= 0.0)
                or np.any(observed <= 0.0)
            ):
                raise PeakedDensityShapeError("nonpositive P4AA measurement")

            observed_normalized = observed / observed[0]
            baseline = np.sqrt(density / density[0])
            candidate_raw = np.sqrt(
                density * np.exp(1.0 - density / peak_density)
            )
            candidate = candidate_raw / candidate_raw[0]
            baseline_error = _relative_error(observed_normalized, baseline)
            candidate_error = _relative_error(observed_normalized, candidate)
            correlation = float(spearmanr(candidate, observed).statistic)
            observed_peak_density = float(density[int(np.argmax(observed))])
            peak_distance = abs(observed_peak_density - peak_density)

            baseline_errors.extend(float(value) for value in baseline_error)
            candidate_errors.extend(float(value) for value in candidate_error)
            candidate_correlations.append(correlation)
            peak_near_count += int(
                peak_distance
                <= config["gates"]["observed_peak_distance_from_2d_maximum"]
            )
            series.append(
                {
                    "table": table,
                    "aperture_um": aperture,
                    "density": density.tolist(),
                    "observed_rms": observed.tolist(),
                    "observed_normalized": observed_normalized.tolist(),
                    "p4d_sqrt_density_normalized": baseline.tolist(),
                    "peaked_density_normalized": candidate.tolist(),
                    "p4d_relative_errors": baseline_error.tolist(),
                    "candidate_relative_errors": candidate_error.tolist(),
                    "candidate_observed_spearman": correlation,
                    "observed_peak_density": observed_peak_density,
                    "observed_peak_distance_from_2d": peak_distance,
                }
            )

    baseline_array = np.asarray(baseline_errors, dtype=np.float64)
    candidate_array = np.asarray(candidate_errors, dtype=np.float64)
    correlation_array = np.asarray(candidate_correlations, dtype=np.float64)
    baseline_median = float(np.median(baseline_array))
    candidate_median = float(np.median(candidate_array))
    baseline_p90 = float(np.quantile(baseline_array, 0.9))
    candidate_p90 = float(np.quantile(candidate_array, 0.9))
    metrics = {
        "eligible_series": len(series),
        "p4d_median_normalized_relative_error": baseline_median,
        "p4d_p90_normalized_relative_error": baseline_p90,
        "candidate_median_normalized_relative_error": candidate_median,
        "candidate_p90_normalized_relative_error": candidate_p90,
        "candidate_median_spearman": float(np.median(correlation_array)),
        "relative_improvement_in_median_error_over_p4d": (
            (baseline_median - candidate_median) / baseline_median
        ),
        "candidate_to_p4d_p90_error_ratio": candidate_p90 / baseline_p90,
        "series_with_observed_peak_near_2d": peak_near_count,
        "finite_positive": bool(
            np.all(np.isfinite(baseline_array))
            and np.all(np.isfinite(candidate_array))
            and np.all(np.isfinite(correlation_array))
            and np.all(baseline_array >= 0.0)
            and np.all(candidate_array >= 0.0)
        ),
    }
    if not math.isfinite(metrics["candidate_median_spearman"]):
        raise PeakedDensityShapeError("undefined P4AC rank statistic")

    gates = config["gates"]
    gate_results = {
        "eligible_series": metrics["eligible_series"]
        == gates["required_eligible_series"],
        "candidate_median_relative_error": metrics[
            "candidate_median_normalized_relative_error"
        ]
        <= gates["candidate_median_normalized_relative_error_maximum"],
        "candidate_p90_relative_error": metrics[
            "candidate_p90_normalized_relative_error"
        ]
        <= gates["candidate_p90_normalized_relative_error_maximum"],
        "candidate_median_spearman": metrics["candidate_median_spearman"]
        >= gates["candidate_median_spearman_minimum"],
        "candidate_median_improvement": metrics[
            "relative_improvement_in_median_error_over_p4d"
        ]
        >= gates["minimum_relative_improvement_in_median_error_over_p4d"],
        "candidate_p90_not_worse": metrics["candidate_to_p4d_p90_error_ratio"]
        <= gates["maximum_candidate_to_p4d_p90_error_ratio"],
        "observed_peak_near_2d": metrics["series_with_observed_peak_near_2d"]
        >= gates["minimum_series_with_peak_near_2d"],
        "finite_positive": metrics["finite_positive"],
        "no_parameter_fit": True,
        "parent_identity": True,
    }
    passed = all(gate_results.values())
    stable = {
        "experiment_id": config["experiment_id"],
        "parent_sha256": identities,
        "model": config["model"],
        "series": series,
        "metrics": metrics,
        "gate_results": gate_results,
        "passed": passed,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            "open_mean_preserving_peaked_compound_poisson_synthetic_leaf"
            if passed
            else "close_peaked_density_shape_without_rescue"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "PeakedDensityShapeError",
    "evaluate_shape",
    "load_contract",
]
