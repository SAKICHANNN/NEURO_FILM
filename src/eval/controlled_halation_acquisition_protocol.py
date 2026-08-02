"""Validate the P3Y controlled-film-halation acquisition protocol."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "neuro_film.u6_p3y_controlled_halation_acquisition_protocol.v1"
REPORT_SCHEMA = "neuro_film.u6_p3y_controlled_halation_acquisition_protocol_report.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HalationAcquisitionProtocolError(ValueError):
    """Raised when the compiled acquisition protocol is invalid or drifts."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _verify_binding(root: Path, binding: dict[str, Any], label: str) -> dict[str, Any]:
    path = root / binding["path"]
    digest = _sha256_bytes(path.read_bytes())
    if digest != binding["sha256"]:
        raise HalationAcquisitionProtocolError(f"P3Y {label} hash drift")
    return (
        json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else {}
    )


def validate_protocol(root: Path, path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    protocol = json.loads(payload.decode("utf-8"))
    if (
        protocol.get("schema") != SCHEMA
        or protocol.get("node") != "ULT > U6 > U6.P3 > U6.P3Y"
    ):
        raise HalationAcquisitionProtocolError("unsupported P3Y protocol")
    if protocol.get("status") != "research-acquisition-protocol-not-data":
        raise HalationAcquisitionProtocolError("P3Y status drift")
    parent = protocol["parent_evidence"]
    _verify_binding(root, parent["p3x_contract"], "parent contract")
    decision = _verify_binding(root, parent["p3x_decision"], "parent decision")
    _verify_binding(root, parent["p3x_evaluator"], "parent evaluator")
    if decision.get("decision") != parent["p3x_decision"]["required_decision"]:
        raise HalationAcquisitionProtocolError("P3Y parent decision drift")
    if not _SHA256.fullmatch(parent["p3x_report_sha256"]):
        raise HalationAcquisitionProtocolError("P3Y parent report identity is invalid")

    roles = protocol["bracket_roles"]
    development_structures = set(roles["development"]["structures"])
    confirmation_structures = set(roles["confirmation"]["structures"])
    structure_overlap = development_structures & confirmation_structures
    development_scales = roles["development"]["exposure_scales"]
    confirmation_scales = roles["confirmation"]["exposure_scales"]
    if len(development_scales) != 3 or len(confirmation_scales) != 3:
        raise HalationAcquisitionProtocolError("P3Y requires two three-exposure roles")
    if structure_overlap:
        raise HalationAcquisitionProtocolError("P3Y role structures overlap")
    hard = protocol["hard_acquisition_requirements"]
    expected_hard = {
        "aligned_source_observations_per_structure": 3,
        "minimum_linear_source_bits": 10,
        "minimum_target_bits": 10,
        "maximum_independent_exposure_anchor_error_ppm": 100,
        "source_clipping_count": 0,
        "target_clipping_count": 0,
        "source_fusion_reads_target": False,
        "fit_reads_confirmation_target": False,
        "development_confirmation_structure_overlap_count": 0,
        "exact_file_and_decoded_sample_hashes_required": True,
        "alignment_evidence_required": True,
        "dark_and_flat_measurements_required": True,
    }
    if hard != expected_hard:
        raise HalationAcquisitionProtocolError("P3Y hard acquisition envelope drift")
    groups = protocol["real_film_group_requirements"]
    if (
        groups["minimum_independent_rolls_per_stock"] < 3
        or groups["minimum_recorded_process_sessions_per_stock"] < 2
        or groups["whole_roll_holdout_required"] is not True
    ):
        raise HalationAcquisitionProtocolError("P3Y stock grouping weakened")
    required_fields = protocol["required_row_fields"]
    if len(required_fields) != len(set(required_fields)) or len(required_fields) < 20:
        raise HalationAcquisitionProtocolError("P3Y row field inventory drift")
    forbidden = set(protocol["forbidden_claims"])
    required_forbidden = {
        "named-stock-calibration-before-held-roll-gates",
        "display-proxy-as-physical-measurement",
        "generic-bloom-as-film-halation",
        "product-promotion",
    }
    if not required_forbidden.issubset(forbidden):
        raise HalationAcquisitionProtocolError("P3Y claim ceiling weakened")
    core = {
        "schema": REPORT_SCHEMA,
        "node": protocol["node"],
        "protocol_sha256": _sha256_bytes(payload),
        "parent_decision": decision["decision"],
        "development_structure_count": len(development_structures),
        "confirmation_structure_count": len(confirmation_structures),
        "structure_overlap_count": len(structure_overlap),
        "development_exposure_count": len(development_scales),
        "confirmation_exposure_count": len(confirmation_scales),
        "required_row_field_count": len(required_fields),
        "minimum_rolls_per_stock": groups["minimum_independent_rolls_per_stock"],
        "minimum_process_sessions_per_stock": groups[
            "minimum_recorded_process_sessions_per_stock"
        ],
        "maximum_anchor_error_ppm": hard[
            "maximum_independent_exposure_anchor_error_ppm"
        ],
        "minimum_source_bits": hard["minimum_linear_source_bits"],
        "minimum_target_bits": hard["minimum_target_bits"],
        "render_authority": False,
        "product_authority": False,
        "passed": True,
        "claim_ceiling": protocol["claim_ceiling"],
    }
    return {**core, "protocol_stable_id": _sha256_bytes(_canonical_bytes(core))}


def validate_candidate_rows(
    protocol: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    required = set(protocol["required_row_fields"])
    if not rows:
        raise HalationAcquisitionProtocolError("P3Y candidate manifest is empty")
    allowed_roles = set(protocol["bracket_roles"])
    for index, row in enumerate(rows):
        if set(row) != required:
            raise HalationAcquisitionProtocolError(
                f"P3Y row {index} field inventory drift"
            )
        if row["role"] not in allowed_roles:
            raise HalationAcquisitionProtocolError(f"P3Y row {index} role is invalid")
        for field in (
            "source_file_sha256",
            "target_file_sha256",
            "source_sample_sha256",
            "target_sample_sha256",
            "exposure_anchor_evidence_sha256",
            "alignment_evidence_sha256",
        ):
            if not isinstance(row[field], str) or not _SHA256.fullmatch(row[field]):
                raise HalationAcquisitionProtocolError(
                    f"P3Y row {index} {field} invalid"
                )
        hard = protocol["hard_acquisition_requirements"]
        if row["source_bits"] < hard["minimum_linear_source_bits"]:
            raise HalationAcquisitionProtocolError(
                f"P3Y row {index} source depth too low"
            )
        if row["target_bits"] < hard["minimum_target_bits"]:
            raise HalationAcquisitionProtocolError(
                f"P3Y row {index} target depth too low"
            )
        if (
            row["exposure_anchor_error_ppm"]
            > hard["maximum_independent_exposure_anchor_error_ppm"]
        ):
            raise HalationAcquisitionProtocolError(f"P3Y row {index} anchor too weak")
        if row["source_clipping_count"] != 0 or row["target_clipping_count"] != 0:
            raise HalationAcquisitionProtocolError(f"P3Y row {index} is clipped")


def write_report(report: dict[str, Any], path: Path) -> str:
    data = _canonical_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _sha256_bytes(data)


__all__ = [
    "HalationAcquisitionProtocolError",
    "validate_candidate_rows",
    "validate_protocol",
    "write_report",
]
