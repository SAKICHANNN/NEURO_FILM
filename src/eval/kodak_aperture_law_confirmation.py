"""U6.P4BF fixed-law confirmation on official Kodak Research tables."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "neuro_film.u6_p4bf_kodak_aperture_law_confirmation_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4bf_kodak_aperture_law_confirmation_report.v1"


class KodakApertureLawConfirmationError(RuntimeError):
    """Raised when frozen P4BF evidence or semantics drift."""


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
        raise KodakApertureLawConfirmationError("P4BF paths must be relative")
    return root / path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate = payload.get("candidate", {})
    controls = payload.get("controls", {})
    evaluation = payload.get("evaluation", {})
    if (
        payload.get("schema") != SCHEMA
        or candidate.get("rms_aperture_law")
        != "sigma(d) = sigma(48um) * 48um / d"
        or candidate.get("fixed_exponent") != 1.0
        or candidate.get("normalization_aperture_micrometres") != 48.0
        or candidate.get("confirmation_apertures_micrometres")
        != [7.25, 12.0, 24.0, 96.0, 192.0, 384.0]
        or candidate.get("normalization_per_density_group") is not True
        or candidate.get("parameter_fit_allowed") is not False
        or candidate.get("row_or_film_omission_allowed") is not False
        or controls.get("constant_rms_exponent") != 0.0
        or controls.get("descriptive_observed_exponent_only") is not True
        or controls.get("selwyn_granularity_column_is_independent_table_check")
        is not True
        or evaluation.get("required_density_group_count") != 23
        or evaluation.get("required_comparison_count") != 138
        or evaluation.get("maximum_median_relative_error") != 0.1
        or evaluation.get("maximum_p90_relative_error") != 0.3
        or evaluation.get("maximum_worst_relative_error") != 0.8
        or evaluation.get("maximum_per_film_median_relative_error") != 0.2
        or evaluation.get("minimum_median_improvement_over_constant") != 0.8
        or evaluation.get("observed_exponent_median_minimum") != 0.9
        or evaluation.get("observed_exponent_median_maximum") != 1.1
        or evaluation.get("maximum_median_selwyn_relative_span") != 0.2
        or evaluation.get("require_two_byte_identical_reports") is not True
    ):
        raise KodakApertureLawConfirmationError("P4BF frozen contract drift")
    return payload


def _load_parent(parents: Mapping[str, Any], root: Path, stem: str) -> dict[str, Any]:
    path = _relative(root, str(parents[f"{stem}_path"]))
    if not path.is_file() or _hash_file(path) != parents[f"{stem}_sha256"]:
        raise KodakApertureLawConfirmationError(
            f"P4BF parent integrity mismatch: {stem}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_confirmation(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    parents = contract["parents"]
    decision = _load_parent(parents, root, "p4be_decision")
    source = _load_parent(parents, root, "p4be_report")
    if (
        decision.get("decision") != "open_fixed_kodak_aperture_law_confirmation"
        or decision.get("source_pass") is not True
        or source.get("stable_evidence_id") != parents["p4be_stable_evidence_id"]
        or source.get("source_pass") is not True
        or source.get("density_group_count") != 23
        or source.get("numeric_measurement_count") != 161
    ):
        raise KodakApertureLawConfirmationError("P4BF parent decision mismatch")

    base_aperture = float(contract["candidate"]["normalization_aperture_micrometres"])
    confirmation_apertures = {
        float(value)
        for value in contract["candidate"]["confirmation_apertures_micrometres"]
    }
    rows: list[dict[str, Any]] = []
    candidate_errors: list[float] = []
    constant_errors: list[float] = []
    exponents: list[float] = []
    selwyn_spans: list[float] = []
    errors_by_film: defaultdict[str, list[float]] = defaultdict(list)
    for group in source["density_groups"]:
        by_aperture = {
            float(item["aperture_diameter_micrometres"]): item
            for item in group["measurements"]
        }
        if set(by_aperture) != confirmation_apertures | {base_aperture}:
            raise KodakApertureLawConfirmationError("P4BF aperture inventory drift")
        base_rms = float(by_aperture[base_aperture]["density_standard_deviation"])
        selwyn = np.asarray(
            [float(item["selwyn_granularity"]) for item in group["measurements"]],
            dtype=np.float64,
        )
        selwyn_spans.append(float((np.max(selwyn) - np.min(selwyn)) / np.median(selwyn)))
        for aperture in sorted(confirmation_apertures):
            observed = float(by_aperture[aperture]["density_standard_deviation"])
            candidate = base_rms * base_aperture / aperture
            constant = base_rms
            candidate_error = abs(candidate / observed - 1.0)
            constant_error = abs(constant / observed - 1.0)
            exponent = -math.log(observed / base_rms) / math.log(
                aperture / base_aperture
            )
            candidate_errors.append(candidate_error)
            constant_errors.append(constant_error)
            exponents.append(exponent)
            errors_by_film[str(group["film"])].append(candidate_error)
            rows.append(
                {
                    "table": group["table"],
                    "film": group["film"],
                    "density": group["density"],
                    "comparison_aperture_micrometres": aperture,
                    "observed_density_standard_deviation": observed,
                    "predicted_density_standard_deviation": candidate,
                    "candidate_relative_error": candidate_error,
                    "constant_control_relative_error": constant_error,
                    "descriptive_observed_exponent": exponent,
                }
            )

    median_error = float(np.median(candidate_errors))
    p90_error = float(np.quantile(candidate_errors, 0.9))
    worst_error = max(candidate_errors)
    median_control = float(np.median(constant_errors))
    median_improvement = 1.0 - median_error / median_control
    median_exponent = float(np.median(exponents))
    per_film_medians = {
        film: float(np.median(values)) for film, values in sorted(errors_by_film.items())
    }
    median_selwyn_span = float(np.median(selwyn_spans))
    gates = contract["evaluation"]
    gate_results = {
        "parent_identity": True,
        "density_group_count": len(source["density_groups"])
        == int(gates["required_density_group_count"]),
        "comparison_count": len(rows) == int(gates["required_comparison_count"]),
        "median_relative_error": median_error
        <= float(gates["maximum_median_relative_error"]),
        "p90_relative_error": p90_error <= float(gates["maximum_p90_relative_error"]),
        "worst_relative_error": worst_error
        <= float(gates["maximum_worst_relative_error"]),
        "per_film_median_relative_error": max(per_film_medians.values())
        <= float(gates["maximum_per_film_median_relative_error"]),
        "improvement_over_constant": median_improvement
        >= float(gates["minimum_median_improvement_over_constant"]),
        "observed_exponent_compatible": median_exponent
        >= float(gates["observed_exponent_median_minimum"])
        and median_exponent <= float(gates["observed_exponent_median_maximum"]),
        "selwyn_column_consistent": median_selwyn_span
        <= float(gates["maximum_median_selwyn_relative_span"]),
        "no_parameter_fit": True,
        "no_row_or_film_omission": True,
    }
    automatic_pass = all(gate_results.values())
    stable = {
        "experiment_id": contract["experiment_id"],
        "density_group_count": len(source["density_groups"]),
        "comparison_count": len(rows),
        "median_candidate_relative_error": median_error,
        "p90_candidate_relative_error": p90_error,
        "worst_candidate_relative_error": worst_error,
        "per_film_median_relative_error": per_film_medians,
        "median_constant_control_relative_error": median_control,
        "median_improvement_over_constant": median_improvement,
        "median_descriptive_observed_exponent": median_exponent,
        "p10_descriptive_observed_exponent": float(np.quantile(exponents, 0.1)),
        "p90_descriptive_observed_exponent": float(np.quantile(exponents, 0.9)),
        "median_selwyn_relative_span": median_selwyn_span,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "rows": rows,
        "decision": (
            "retain_historical_photographic_aperture_law_compatibility"
            if automatic_pass
            else "close_independent_cell_spatial_interpretation_across_sources"
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
    "KodakApertureLawConfirmationError",
    "evaluate_confirmation",
    "load_contract",
    "write_report",
]
