"""Frozen external-amplitude check for the unchanged U6.P6N Callier profile."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_callier_operator import (
    load_contract as load_operator_contract,
    profile_from_contract,
)
from src.eval.physical_callier_source import hash_file
from src.film_physics.callier import callier_q_factor


SCHEMA = "neuro_film.u6_p6p_fixed_callier_measurement_compatibility_contract.v1"
REPORT_SCHEMA = (
    "neuro_film.u6_p6p_fixed_callier_measurement_compatibility_report.v1"
)


class CallierMeasurementCompatibilityError(RuntimeError):
    """Raised when the P6P contract or its frozen parents drift."""


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
        or evaluation.get("collimation") != 1.0
        or evaluation.get("silver_samples") != ["silverHD", "silverLD"]
        or evaluation.get("dye_samples") != ["dyeHD", "dyeLD"]
        or evaluation.get("runs") != 2
        or evaluation.get("post_result_retuning_allowed")
        or gates.get("silver_model_envelope_absolute_tolerance") != 0.05
        or gates.get("dye_reported_q_distance_from_identity_maximum") != 0.1
        or gates.get("dye_raw_ratio_distance_from_identity_maximum") != 0.1
        or not gates.get("all_silver_reported_q_must_be_within_envelope")
        or not gates.get("all_silver_raw_ratio_must_be_within_envelope")
        or not gates.get("silver_q_must_exceed_same_level_dye_q")
        or not gates.get("no_parameter_fit")
    ):
        raise CallierMeasurementCompatibilityError("P6P frozen contract drift")
    return payload


def _validated_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise CallierMeasurementCompatibilityError("P6P parent path must be relative")
    return root / path


def _load_frozen_parents(
    config: dict[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    parents = config["parents"]
    identities: dict[str, str] = {}
    for stem in ("p6n_contract", "p6n_decision", "p6o_contract", "p6o_report"):
        path = _validated_path(root, parents[f"{stem}_path"])
        if not path.is_file():
            raise CallierMeasurementCompatibilityError(f"missing P6P parent: {stem}")
        actual = hash_file(path)
        expected = parents[f"{stem}_sha256"]
        if actual != expected:
            raise CallierMeasurementCompatibilityError(
                f"P6P parent hash mismatch: {stem}"
            )
        identities[stem] = actual
    operator = load_operator_contract(
        _validated_path(root, parents["p6n_contract_path"])
    )
    report = json.loads(
        _validated_path(root, parents["p6o_report_path"]).read_text(encoding="utf-8")
    )
    if (
        report.get("schema")
        != "neuro_film.u6_p6o_callier_measurement_source_report.v1"
        or report.get("stable_evidence_id") != parents["p6o_stable_evidence_id"]
        or not report.get("source_pass")
    ):
        raise CallierMeasurementCompatibilityError("P6O evidence identity mismatch")
    return operator, report, identities


def _within(value: float, low: float, high: float, tolerance: float) -> bool:
    return low - tolerance <= value <= high + tolerance


def evaluate_compatibility(config: dict[str, Any], root: Path) -> dict[str, Any]:
    operator, source_report, identities = _load_frozen_parents(config, root)
    profile = profile_from_contract(operator)
    gates = config["gates"]
    tolerance = float(gates["silver_model_envelope_absolute_tolerance"])
    rows = {row["sample_id"]: row for row in source_report["measurements"]}
    expected_ids = set(config["evaluation"]["silver_samples"]) | set(
        config["evaluation"]["dye_samples"]
    )
    if set(rows) != expected_ids:
        raise CallierMeasurementCompatibilityError("P6O measurement row set drift")

    comparisons = []
    for sample_id in config["evaluation"]["silver_samples"]:
        row = rows[sample_id]
        density = np.full((1, 1, 3), row["diffuse_density"], dtype=np.float64)
        q_rgb = callier_q_factor(
            density, profile, collimation=config["evaluation"]["collimation"]
        )[0, 0]
        low = float(np.min(q_rgb))
        high = float(np.max(q_rgb))
        comparisons.append(
            {
                "sample_id": sample_id,
                "diffuse_density": row["diffuse_density"],
                "model_q_rgb": [float(value) for value in q_rgb],
                "model_envelope": [low, high],
                "reported_q": row["reported_q"],
                "raw_density_ratio": row["raw_density_ratio"],
                "reported_within_envelope": _within(
                    row["reported_q"], low, high, tolerance
                ),
                "raw_ratio_within_envelope": _within(
                    row["raw_density_ratio"], low, high, tolerance
                ),
            }
        )

    dye_controls = []
    for sample_id in config["evaluation"]["dye_samples"]:
        row = rows[sample_id]
        dye_controls.append(
            {
                "sample_id": sample_id,
                "density_level": row["density_level"],
                "reported_q": row["reported_q"],
                "raw_density_ratio": row["raw_density_ratio"],
                "reported_distance_from_identity": abs(row["reported_q"] - 1.0),
                "raw_ratio_distance_from_identity": abs(
                    row["raw_density_ratio"] - 1.0
                ),
            }
        )

    finite_values = [
        value
        for row in rows.values()
        for value in (
            row["directed_density"],
            row["diffuse_density"],
            row["reported_q"],
            row["raw_density_ratio"],
        )
    ]
    level_pairs = [
        (rows["silverHD"], rows["dyeHD"]),
        (rows["silverLD"], rows["dyeLD"]),
    ]
    gate_results = {
        "parent_identity": True,
        "finite_measurements": all(math.isfinite(value) for value in finite_values),
        "silver_reported_envelope": all(
            row["reported_within_envelope"] for row in comparisons
        ),
        "silver_raw_ratio_envelope": all(
            row["raw_ratio_within_envelope"] for row in comparisons
        ),
        "dye_reported_identity": all(
            row["reported_distance_from_identity"]
            <= gates["dye_reported_q_distance_from_identity_maximum"]
            for row in dye_controls
        ),
        "dye_raw_ratio_identity": all(
            row["raw_ratio_distance_from_identity"]
            <= gates["dye_raw_ratio_distance_from_identity_maximum"]
            for row in dye_controls
        ),
        "silver_exceeds_dye_same_level": all(
            silver["reported_q"] > dye["reported_q"]
            and silver["raw_density_ratio"] > dye["raw_density_ratio"]
            for silver, dye in level_pairs
        ),
        "no_parameter_fit": True,
    }
    passed = all(gate_results.values())
    stable = {
        "experiment_id": config["experiment_id"],
        "parent_sha256": identities,
        "profile": operator["operator"]["profile"],
        "silver_comparisons": comparisons,
        "dye_controls": dye_controls,
        "reported_vs_raw_maximum_difference": source_report[
            "maximum_reported_vs_raw_ratio_difference"
        ],
        "gate_results": gate_results,
        "passed": passed,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            "retain_fixed_p6n_quantitative_compatibility"
            if passed
            else "close_fixed_p6n_quantitative_compatibility_without_retuning"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "CallierMeasurementCompatibilityError",
    "evaluate_compatibility",
    "load_contract",
]
