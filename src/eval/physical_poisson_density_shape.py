"""No-fit measured shape audit for the unchanged U6.P4D Poisson model."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from src.eval.physical_callier_source import hash_file


SCHEMA = "neuro_film.u6_p4ab_fixed_poisson_density_shape_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4ab_fixed_poisson_density_shape_report.v1"


class PoissonDensityShapeError(RuntimeError):
    """Raised when the P4AB contract or frozen evidence drifts."""


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    evaluation = payload.get("evaluation", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or evaluation.get("tables") != [4, 5, 6]
        or evaluation.get("apertures_um") != [13, 27, 57]
        or evaluation.get("minimum_points_per_series") != 4
        or evaluation.get("runs") != 2
        or evaluation.get("post_result_retuning_allowed")
        or gates.get("required_eligible_series") != 9
        or gates.get("median_normalized_relative_error_maximum") != 0.2
        or gates.get("p90_normalized_relative_error_maximum") != 0.35
        or gates.get("median_spearman_minimum") != 0.7
        or gates.get("minimum_series_with_observed_maximum_at_highest_density")
        != 7
        or not gates.get("no_parameter_fit")
    ):
        raise PoissonDensityShapeError("P4AB frozen contract drift")
    return payload


def _relative(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise PoissonDensityShapeError("P4AB parent path must be relative")
    return root / path


def _load_parents(
    config: dict[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    parents = config["parents"]
    identities = {}
    for stem in ("p4d_contract", "p4d_decision", "p4aa_report"):
        path = _relative(root, parents[f"{stem}_path"])
        if not path.is_file():
            raise PoissonDensityShapeError(f"missing P4AB parent: {stem}")
        actual = hash_file(path)
        if actual != parents[f"{stem}_sha256"]:
            raise PoissonDensityShapeError(f"P4AB parent hash mismatch: {stem}")
        identities[stem] = actual
    p4d = json.loads(
        _relative(root, parents["p4d_contract_path"]).read_text(encoding="utf-8")
    )
    if (
        p4d.get("schema")
        != "neuro_film.u6_p4d_density_conditioned_structure_contract.v1"
        or p4d.get("model", {}).get("expected_grain_count_per_pixel")
        != "target_density / grain_optical_density"
        or p4d.get("model", {}).get("grain_count_distribution")
        != "coordinate-counter Poisson"
    ):
        raise PoissonDensityShapeError("P4D model identity mismatch")
    source = json.loads(
        _relative(root, parents["p4aa_report_path"]).read_text(encoding="utf-8")
    )
    if (
        source.get("schema")
        != "neuro_film.u6_p4aa_nasa_density_grain_source_report.v1"
        or source.get("stable_evidence_id") != parents["p4aa_stable_evidence_id"]
        or not source.get("source_pass")
    ):
        raise PoissonDensityShapeError("P4AA evidence identity mismatch")
    return p4d, source, identities


def evaluate_shape(config: dict[str, Any], root: Path) -> dict[str, Any]:
    p4d, source, identities = _load_parents(config, root)
    rows = source["numeric_density_granularity_rows"]
    series = []
    all_errors: list[float] = []
    correlations: list[float] = []
    turnover_count = 0
    minimum_points = int(config["evaluation"]["minimum_points_per_series"])
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
                raise PoissonDensityShapeError("nonpositive P4AA measurement")
            predicted = np.sqrt(density / density[0])
            observed_normalized = observed / observed[0]
            relative_error = np.abs(observed_normalized - predicted) / predicted
            correlation = float(spearmanr(density, observed).statistic)
            maximum_at_highest = bool(np.argmax(observed) == len(observed) - 1)
            all_errors.extend(float(value) for value in relative_error)
            correlations.append(correlation)
            turnover_count += int(maximum_at_highest)
            series.append(
                {
                    "table": table,
                    "aperture_um": aperture,
                    "density": density.tolist(),
                    "observed_rms": observed.tolist(),
                    "observed_normalized": observed_normalized.tolist(),
                    "p4d_sqrt_density_normalized": predicted.tolist(),
                    "relative_errors": relative_error.tolist(),
                    "spearman": correlation,
                    "observed_maximum_at_highest_density": maximum_at_highest,
                }
            )
    errors = np.asarray(all_errors, dtype=np.float64)
    correlations_array = np.asarray(correlations, dtype=np.float64)
    gates = config["gates"]
    metrics = {
        "eligible_series": len(series),
        "median_normalized_relative_error": float(np.median(errors)),
        "p90_normalized_relative_error": float(np.quantile(errors, 0.9)),
        "median_spearman": float(np.median(correlations_array)),
        "series_with_observed_maximum_at_highest_density": turnover_count,
        "finite_positive": bool(
            np.all(np.isfinite(errors))
            and np.all(np.isfinite(correlations_array))
            and np.all(errors >= 0.0)
        ),
    }
    gate_results = {
        "eligible_series": metrics["eligible_series"]
        == gates["required_eligible_series"],
        "median_relative_error": metrics["median_normalized_relative_error"]
        <= gates["median_normalized_relative_error_maximum"],
        "p90_relative_error": metrics["p90_normalized_relative_error"]
        <= gates["p90_normalized_relative_error_maximum"],
        "median_spearman": metrics["median_spearman"]
        >= gates["median_spearman_minimum"],
        "monotone_high_density_shape": metrics[
            "series_with_observed_maximum_at_highest_density"
        ]
        >= gates["minimum_series_with_observed_maximum_at_highest_density"],
        "finite_positive": metrics["finite_positive"],
        "no_parameter_fit": True,
        "parent_identity": True,
    }
    passed = all(gate_results.values())
    stable = {
        "experiment_id": config["experiment_id"],
        "parent_sha256": identities,
        "p4d_model": {
            "expected_grain_count_per_pixel": p4d["model"][
                "expected_grain_count_per_pixel"
            ],
            "grain_count_distribution": p4d["model"]["grain_count_distribution"],
            "implied_rms_shape": "sqrt(density)",
        },
        "series": series,
        "metrics": metrics,
        "gate_results": gate_results,
        "passed": passed,
    }
    if not math.isfinite(metrics["median_spearman"]):
        raise PoissonDensityShapeError("undefined P4AB rank statistic")
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            "retain_p4d_generic_measured_shape_control"
            if passed
            else "close_p4d_measured_shape_compatibility_without_retuning"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "PoissonDensityShapeError",
    "evaluate_shape",
    "load_contract",
]
