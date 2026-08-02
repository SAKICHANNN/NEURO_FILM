"""U6.P4BD no-fit measured aperture-scaling compatibility evaluator."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "neuro_film.u6_p4bd_aperture_scaling_compatibility_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4bd_aperture_scaling_compatibility_report.v1"


class ApertureScalingCompatibilityError(RuntimeError):
    """Raised when frozen P4BD evidence or semantics drift."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ApertureScalingCompatibilityError("P4BD paths must be relative")
    return root / path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate = payload.get("candidate", {})
    controls = payload.get("controls", {})
    evaluation = payload.get("evaluation", {})
    if (
        payload.get("schema") != SCHEMA
        or candidate.get("mechanism")
        != "independent_two_dimensional_measurement_cells"
        or candidate.get("rms_aperture_law")
        != "sigma(d) = sigma(13um) * 13um / d"
        or candidate.get("fixed_exponent") != 1.0
        or candidate.get("normalization_aperture_micrometres") != 13
        or candidate.get("comparison_apertures_micrometres") != [27, 57]
        or candidate.get("parameter_fit_allowed") is not False
        or candidate.get("missing_value_imputation_allowed") is not False
        or controls.get("constant_rms_exponent") != 0.0
        or controls.get("descriptive_observed_exponent_only") is not True
        or controls.get("descriptive_exponent_may_not_replace_candidate") is not True
        or evaluation.get("minimum_complete_comparisons") != 20
        or evaluation.get("maximum_median_relative_error") != 0.2
        or evaluation.get("maximum_worst_relative_error") != 0.4
        or evaluation.get("minimum_median_improvement_over_constant") != 0.3
        or evaluation.get("observed_exponent_median_minimum") != 0.8
        or evaluation.get("observed_exponent_median_maximum") != 1.2
        or evaluation.get("require_two_byte_identical_reports") is not True
    ):
        raise ApertureScalingCompatibilityError("P4BD frozen contract drift")
    return payload


def _load_parent(parents: Mapping[str, Any], root: Path, stem: str) -> dict[str, Any]:
    path = _relative(root, str(parents[f"{stem}_path"]))
    if not path.is_file() or _hash_file(path) != parents[f"{stem}_sha256"]:
        raise ApertureScalingCompatibilityError(
            f"P4BD parent integrity mismatch: {stem}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_compatibility(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    parents = contract["parents"]
    p4bc = _load_parent(parents, root, "p4bc_decision")
    p4aa_contract = _load_parent(parents, root, "p4aa_contract")
    source = _load_parent(parents, root, "p4aa_report")
    if (
        p4bc.get("decision") != "retain_generic_aperture_cell_reference_sampler"
        or p4bc.get("automatic_pass") is not True
        or p4aa_contract.get("source_gate", {}).get("no_parameter_fit") is not True
        or source.get("source_pass") is not True
        or source.get("stable_evidence_id") != parents["p4aa_stable_evidence_id"]
        or source.get("numeric_density_granularity_row_count") != 14
    ):
        raise ApertureScalingCompatibilityError("P4BD parent decision mismatch")

    base_aperture = float(contract["candidate"]["normalization_aperture_micrometres"])
    comparison_apertures = {
        float(value) for value in contract["candidate"]["comparison_apertures_micrometres"]
    }
    rows: list[dict[str, Any]] = []
    candidate_errors: list[float] = []
    constant_errors: list[float] = []
    observed_exponents: list[float] = []
    for source_row in source["numeric_density_granularity_rows"]:
        values = {
            float(key): float(value)
            for key, value in source_row["rms_granularity_by_aperture_um"].items()
        }
        if base_aperture not in values:
            continue
        base_rms = values[base_aperture]
        for aperture in sorted(comparison_apertures & values.keys()):
            observed = values[aperture]
            candidate = base_rms * base_aperture / aperture
            constant = base_rms
            candidate_error = abs(candidate / observed - 1.0)
            constant_error = abs(constant / observed - 1.0)
            exponent = -math.log(observed / base_rms) / math.log(
                aperture / base_aperture
            )
            candidate_errors.append(candidate_error)
            constant_errors.append(constant_error)
            observed_exponents.append(exponent)
            rows.append(
                {
                    "table": source_row["table"],
                    "row": source_row["row"],
                    "density": source_row["density"],
                    "base_aperture_micrometres": base_aperture,
                    "comparison_aperture_micrometres": aperture,
                    "base_rms_granularity": base_rms,
                    "observed_rms_granularity": observed,
                    "independent_cell_predicted_rms": candidate,
                    "independent_cell_relative_error": candidate_error,
                    "constant_control_relative_error": constant_error,
                    "descriptive_observed_exponent": exponent,
                }
            )

    median_candidate_error = float(np.median(candidate_errors))
    worst_candidate_error = max(candidate_errors)
    median_constant_error = float(np.median(constant_errors))
    median_improvement = 1.0 - median_candidate_error / median_constant_error
    median_exponent = float(np.median(observed_exponents))
    gates = contract["evaluation"]
    gate_results = {
        "parent_identity": True,
        "complete_comparisons": len(rows)
        >= int(gates["minimum_complete_comparisons"]),
        "median_relative_error": median_candidate_error
        <= float(gates["maximum_median_relative_error"]),
        "worst_relative_error": worst_candidate_error
        <= float(gates["maximum_worst_relative_error"]),
        "improvement_over_constant": median_improvement
        >= float(gates["minimum_median_improvement_over_constant"]),
        "observed_exponent_compatible": median_exponent
        >= float(gates["observed_exponent_median_minimum"])
        and median_exponent <= float(gates["observed_exponent_median_maximum"]),
        "no_parameter_fit": True,
        "no_missing_value_imputation": True,
    }
    automatic_pass = all(gate_results.values())
    stable = {
        "experiment_id": contract["experiment_id"],
        "complete_comparison_count": len(rows),
        "median_independent_cell_relative_error": median_candidate_error,
        "worst_independent_cell_relative_error": worst_candidate_error,
        "median_constant_control_relative_error": median_constant_error,
        "median_improvement_over_constant": median_improvement,
        "median_descriptive_observed_exponent": median_exponent,
        "p10_descriptive_observed_exponent": float(
            np.quantile(observed_exponents, 0.1)
        ),
        "p90_descriptive_observed_exponent": float(
            np.quantile(observed_exponents, 0.9)
        ),
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "rows": rows,
        "decision": (
            "retain_cross_material_independent_cell_aperture_compatibility"
            if automatic_pass
            else "close_independent_cell_spatial_interpretation"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }


def write_report(report: Mapping[str, Any], path: Path) -> str:
    encoded = _canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "ApertureScalingCompatibilityError",
    "evaluate_compatibility",
    "load_contract",
    "write_report",
]
