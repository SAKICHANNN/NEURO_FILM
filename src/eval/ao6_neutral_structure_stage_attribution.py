"""P4IQ stage attribution for the closed P4IP neutral structure."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.layer_gamma_photographic_development import _high_frequency_chroma_p999
from src.eval.neutral_transmittance_structure_ao6_d0 import (
    apply_neutral_transmittance_structure,
)
from src.eval.physical_spatial_photographic_stress import _isolated_excursions
from src.eval.sigmoid_scanner_ao6_value_d1 import evaluate as evaluate_base
from src.eval.sigmoid_scanner_ao6_value_d1 import load_contract as load_base_contract

SCHEMA = "neuro-film.u6-p4iq-ao6-stage-attribution-contract.v1"
REPORT_SCHEMA = "neuro-film.u6-p4iq-ao6-stage-attribution-result.v1"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("stages") != [
        "scanner_encoded",
        "ao6_base",
        "ao6_residual",
    ]:
        raise ValueError("unsupported P4IQ contract")
    return payload


def evaluate(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    parent_lock = contract["parents"]["p4ip_evidence"]
    parent_path = root / parent_lock["path"]
    if _sha(parent_path) != parent_lock["sha256"]:
        raise ValueError("P4IQ parent identity drift")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("decision") != parent_lock["required_decision"]:
        raise ValueError("P4IQ parent decision drift")
    base_lock = contract["parents"]["base_contract"]
    base_path = root / base_lock["path"]
    if _sha(base_path) != base_lock["sha256"]:
        raise ValueError("P4IQ base contract drift")
    threshold = contract["diagnostic_thresholds"]
    rows: list[dict[str, Any]] = []

    def observe(source_id: str, values: Mapping[str, np.ndarray]) -> None:
        pairs = {
            "scanner_encoded": (values["baseline_encoded"], values["physical_encoded"]),
            "ao6_base": (values["matched_base"], values["physical_base"]),
            "ao6_residual": (values["matched_output"], values["physical_output"]),
        }
        stages: dict[str, Any] = {}
        for stage, (baseline, candidate) in pairs.items():
            delta = candidate.astype(np.float64) - baseline.astype(np.float64)
            stages[stage] = {
                "p95_abs": float(np.percentile(np.abs(delta), 95)),
                "p99_abs": float(np.percentile(np.abs(delta), 99)),
                "high_frequency_chroma_p999": _high_frequency_chroma_p999(delta),
                "isolated_excursions": _isolated_excursions(
                    delta,
                    threshold=float(threshold["isolated_excursion_threshold"]),
                    radius=int(threshold["isolated_support_radius_pixels"]),
                    minimum_support=int(threshold["minimum_isolated_support_count"]),
                ),
            }
        rows.append({"id": source_id, "stages": stages})

    evaluate_base(
        load_base_contract(base_path),
        root,
        scan_structure_builder=apply_neutral_transmittance_structure,
        stage_observer=observe,
    )
    aggregate = {
        stage: {
            "maximum_high_frequency_chroma_p999": max(
                row["stages"][stage]["high_frequency_chroma_p999"] for row in rows
            ),
            "total_isolated_excursions": sum(
                row["stages"][stage]["isolated_excursions"] for row in rows
            ),
            "population_p95_abs": float(
                np.percentile([row["stages"][stage]["p95_abs"] for row in rows], 95)
            ),
            "population_p99_abs": float(
                np.percentile([row["stages"][stage]["p99_abs"] for row in rows], 99)
            ),
        }
        for stage in contract["stages"]
    }
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(),
        "parent_stable_evidence_id": parent["stable_evidence_id"],
        "rows": rows,
        "aggregate": aggregate,
        "decision": contract["decision"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
