"""Validate filled SF3.A0N capture-condition and exposure receipts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA = "neuro-film.sf3-a0n-three-stock-capture-receipt-contract.v1"
PACKET_SCHEMA = "neuro-film.sf3-a0n-three-stock-capture-receipt-packet.v1"
REPORT_SCHEMA = "neuro-film.sf3-a0n-three-stock-capture-receipt-report.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ThreeStockCaptureReceiptError(ValueError):
    """Raised when a capture receipt packet violates the frozen work order."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _read_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ThreeStockCaptureReceiptError(f"expected JSON object: {path}")
    return raw, value


def _bound_file(root: Path, binding: dict[str, Any], label: str) -> Path:
    relative = Path(str(binding.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ThreeStockCaptureReceiptError(f"invalid {label} path")
    path = root.joinpath(*relative.parts)
    if not path.is_file() or _sha256(path.read_bytes()) != binding.get("sha256"):
        raise ThreeStockCaptureReceiptError(f"{label} identity drift")
    return path


def _finite_number(value: Any, *, positive: bool = False) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and (not positive or float(value) > 0.0)
    )


def _load_contract_and_work_order(
    contract_path: Path, *, root: Path
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    contract_raw, contract = _read_object(contract_path)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != "FROZEN_BEFORE_PHYSICAL_CAPTURE"
    ):
        raise ThreeStockCaptureReceiptError("unsupported or unfrozen SF3.A0N contract")
    work_order_path = _bound_file(root, contract["parents"]["work_order"], "work order")
    _, work_order = _read_object(work_order_path)
    if (
        work_order.get("stable_evidence_id")
        != contract["parents"]["work_order"]["stable_evidence_id"]
    ):
        raise ThreeStockCaptureReceiptError("work-order stable identity drift")
    return contract_raw, contract, work_order


def build_receipt_template(contract_path: Path, *, root: Path) -> dict[str, Any]:
    """Build the exact fillable packet shape without inventing measurements."""

    _, _, work_order = _load_contract_and_work_order(contract_path, root=root)
    conditions = []
    for expected in work_order["common_condition_records"]:
        row = {field: None for field in expected["required_fields"]}
        row["condition_slot_id"] = expected["condition_slot_id"]
        row["stimulus_sha256"] = expected["stimulus_sha256"]
        conditions.append(row)
    exposures = []
    for expected in work_order["exposure_rows"]:
        row = {field: None for field in expected["exposure_receipt_required_fields"]}
        row.update(
            {
                "exposure_slot_id": expected["exposure_slot_id"],
                "stock_id": expected["stock_id"],
                "film_ei": expected["nominal_iso"],
                "nominal_iso": expected["nominal_iso"],
            }
        )
        exposures.append(row)
    return {
        "schema": PACKET_SCHEMA,
        "work_order_stable_evidence_id": work_order["stable_evidence_id"],
        "common_condition_records": conditions,
        "exposure_receipts": exposures,
    }


def evaluate_receipts(
    contract_path: Path, packet_path: Path, *, root: Path
) -> dict[str, Any]:
    contract_raw, contract, work_order = _load_contract_and_work_order(
        contract_path, root=root
    )

    packet_raw, packet = _read_object(packet_path)
    if set(packet) != {
        "schema",
        "work_order_stable_evidence_id",
        "common_condition_records",
        "exposure_receipts",
    }:
        raise ThreeStockCaptureReceiptError("capture receipt packet field drift")
    if (
        packet.get("schema") != PACKET_SCHEMA
        or packet.get("work_order_stable_evidence_id")
        != work_order["stable_evidence_id"]
    ):
        raise ThreeStockCaptureReceiptError("capture receipt packet parent drift")

    expected_conditions = {
        row["condition_slot_id"]: row for row in work_order["common_condition_records"]
    }
    expected_exposures = {
        row["exposure_slot_id"]: row for row in work_order["exposure_rows"]
    }
    conditions = packet.get("common_condition_records")
    exposures = packet.get("exposure_receipts")
    if not isinstance(conditions, list) or not isinstance(exposures, list):
        raise ThreeStockCaptureReceiptError("capture receipt inventories are invalid")
    if (
        len(conditions) != contract["expected_counts"]["common_condition_records"]
        or len(exposures) != contract["expected_counts"]["exposure_receipts"]
    ):
        raise ThreeStockCaptureReceiptError("capture receipt count drift")

    condition_by_id: dict[str, dict[str, Any]] = {}
    for row in conditions:
        slot = row.get("condition_slot_id") if isinstance(row, dict) else None
        expected = expected_conditions.get(slot)
        required = set(expected["required_fields"]) if expected else set()
        if (
            expected is None
            or set(row) != {"condition_slot_id", *required}
            or slot in condition_by_id
        ):
            raise ThreeStockCaptureReceiptError("common-condition slot or field drift")
        if row["stimulus_sha256"] != expected["stimulus_sha256"]:
            raise ThreeStockCaptureReceiptError("stimulus identity drift")
        for key in ("display_device_id", "camera_system_id", "lens_id", "framing_id"):
            if not isinstance(row[key], str) or not row[key]:
                raise ThreeStockCaptureReceiptError(f"invalid common-condition {key}")
        if not _SHA256.fullmatch(str(row["display_profile_sha256"])):
            raise ThreeStockCaptureReceiptError("invalid display profile identity")
        for key in ("display_luminance_cd_m2", "aperture_f_number", "focus_distance_m"):
            if not _finite_number(row[key], positive=True):
                raise ThreeStockCaptureReceiptError(f"invalid common-condition {key}")
        if (
            not _finite_number(row["ambient_illuminance_lux"])
            or row["ambient_illuminance_lux"] < 0
        ):
            raise ThreeStockCaptureReceiptError("invalid ambient illuminance")
        condition_by_id[slot] = row

    exposure_by_id: dict[str, dict[str, Any]] = {}
    for row in exposures:
        slot = row.get("exposure_slot_id") if isinstance(row, dict) else None
        expected = expected_exposures.get(slot)
        required = (
            set(expected["exposure_receipt_required_fields"]) if expected else set()
        )
        if (
            expected is None
            or set(row) != {"exposure_slot_id", *required}
            or slot in exposure_by_id
        ):
            raise ThreeStockCaptureReceiptError("exposure slot or field drift")
        if (
            row["stock_id"] != expected["stock_id"]
            or row["nominal_iso"] != expected["nominal_iso"]
            or row["film_ei"] != expected["nominal_iso"]
        ):
            raise ThreeStockCaptureReceiptError("stock or box-speed exposure drift")
        if not _finite_number(row["shutter_seconds"], positive=True):
            raise ThreeStockCaptureReceiptError("invalid shutter duration")
        for key in ("meter_reading_ev100", "exposure_compensation_ev"):
            if not _finite_number(row[key]):
                raise ThreeStockCaptureReceiptError(f"invalid exposure {key}")
        if (
            not isinstance(row["meter_id"], str)
            or not row["meter_id"]
            or not _SHA256.fullmatch(str(row["meter_calibration_sha256"]))
        ):
            raise ThreeStockCaptureReceiptError("invalid exposure meter identity")
        exposure_by_id[slot] = row

    if set(condition_by_id) != set(expected_conditions) or set(exposure_by_id) != set(
        expected_exposures
    ):
        raise ThreeStockCaptureReceiptError("capture receipt slot coverage drift")
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": _sha256(contract_raw),
        "work_order_stable_evidence_id": work_order["stable_evidence_id"],
        "packet_sha256": _sha256(packet_raw),
        "common_condition_records": len(conditions),
        "exposure_receipts": len(exposures),
        "automatic_pass": True,
        "decision": contract["decision_if_pass"],
        "operator_fits": 0,
        "film_target_scores": 0,
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": _sha256(_canonical(core))}


__all__ = [
    "ThreeStockCaptureReceiptError",
    "build_receipt_template",
    "evaluate_receipts",
]
