"""Validate filled SF3.A0N capture-condition and exposure receipts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from src.real_film.three_stock_acquisition import (
    LEDGER_SCHEMA,
    compile_acquisition_ledger,
)

CONTRACT_SCHEMA = "neuro-film.sf3-a0n-three-stock-capture-receipt-contract.v1"
PACKET_SCHEMA = "neuro-film.sf3-a0n-three-stock-capture-receipt-packet.v1"
REPORT_SCHEMA = "neuro-film.sf3-a0n-three-stock-capture-receipt-report.v1"
BINDING_REPORT_SCHEMA = (
    "neuro-film.sf3-a0n-three-stock-receipt-ledger-binding-report.v1"
)
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


def build_ledger_template(
    contract_path: Path,
    packet_path: Path,
    acquisition_contract_path: Path,
    *,
    root: Path,
) -> dict[str, Any]:
    """Build the exact 108-row evidence-ledger skeleton from verified receipts."""

    evaluate_receipts(contract_path, packet_path, root=root)
    _, _, work_order = _load_contract_and_work_order(contract_path, root=root)
    _, packet = _read_object(packet_path)
    acquisition_raw, acquisition_contract = _read_object(acquisition_contract_path)
    if (
        acquisition_contract.get("schema")
        != "neuro-film.sf3-a0-three-stock-controlled-acquisition-contract.v1"
        or _sha256(acquisition_raw) != work_order["acquisition_contract_sha256"]
    ):
        raise ThreeStockCaptureReceiptError("acquisition contract identity drift")
    condition_by_id = {
        row["condition_slot_id"]: row for row in work_order["common_condition_records"]
    }
    receipt_condition_by_id = {
        row["condition_slot_id"]: row for row in packet["common_condition_records"]
    }
    exposure_by_id = {
        row["exposure_slot_id"]: row for row in work_order["exposure_rows"]
    }
    path_fields = {
        "digital_reference_sha256": "digital_reference_path",
        "capture_condition_sha256": "capture_condition_record_path",
        "process_recipe_sha256": "process_recipe_record_path",
        "scanner_profile_sha256": "scanner_profile_record_path",
        "scan_file_sha256": "scan_file_path",
        "scan_sample_sha256": "scan_sample_path",
        "alignment_evidence_sha256": "alignment_evidence_path",
        "rights_record_sha256": "rights_record_path",
    }
    ledger_fields = (
        set(acquisition_contract["required_row_fields"]) - set(path_fields)
    ) | set(path_fields.values())
    rows: list[dict[str, Any]] = []
    for task in work_order["scan_tasks"]:
        if task["counts_toward_evidence_minimum"] is not True:
            continue
        exposure = exposure_by_id[task["exposure_slot_id"]]
        condition = condition_by_id[exposure["common_condition_slot_id"]]
        receipt_condition = receipt_condition_by_id[condition["condition_slot_id"]]
        row = {field: None for field in ledger_fields}
        row.update(
            {
                "row_id": task["scan_task_id"],
                "stock_id": exposure["stock_id"],
                "role": exposure["role"],
                "scene_id": condition["scene_id"],
                "film_frame_id": exposure["exposure_slot_id"],
                "camera_system_id": receipt_condition["camera_system_id"],
                "roll_id": exposure["roll_slot_id"],
                "process_session_id": exposure["process_session_slot_id"],
                "process_type": exposure["process_type"],
                "scanner_session_id": task["scanner_session_slot_id"],
                "scanner_device_id": task["scanner_device_slot_id"],
                "interpretation_id": exposure["interpretation_id"],
            }
        )
        rows.append(row)
    expected = int(work_order["counts"]["evidence_scan_tasks"])
    if len(rows) != expected or any(set(row) != ledger_fields for row in rows):
        raise ThreeStockCaptureReceiptError("ledger template inventory drift")
    return {"schema": LEDGER_SCHEMA, "rows": rows}


def evaluate_ledger_binding(
    contract_path: Path,
    packet_path: Path,
    acquisition_contract_path: Path,
    ledger_path: Path,
    *,
    root: Path,
) -> dict[str, Any]:
    """Bind one passing receipt packet to the exact evidence scan-task ledger.

    This is deliberately pre-pixel plumbing.  Diagnostic exposures remain in the
    physical work order but do not enter the SF3.A0 evidence ledger.
    """

    receipt_report = evaluate_receipts(contract_path, packet_path, root=root)
    _, _, work_order = _load_contract_and_work_order(contract_path, root=root)
    packet_raw, packet = _read_object(packet_path)
    ledger_raw, ledger = _read_object(ledger_path)
    acquisition_contract_raw = acquisition_contract_path.read_bytes()
    manifest = compile_acquisition_ledger(
        acquisition_contract_path, ledger_path, root=root
    )

    source_rows = ledger.get("rows")
    if not isinstance(source_rows, list):
        raise ThreeStockCaptureReceiptError("invalid acquisition ledger rows")
    manifest_rows = manifest.get("rows")
    if not isinstance(manifest_rows, list) or len(manifest_rows) != len(source_rows):
        raise ThreeStockCaptureReceiptError("compiled manifest row-count drift")

    exposure_by_id = {
        row["exposure_slot_id"]: row for row in work_order["exposure_rows"]
    }
    condition_by_id = {
        row["condition_slot_id"]: row for row in work_order["common_condition_records"]
    }
    receipt_condition_by_id = {
        row["condition_slot_id"]: row for row in packet["common_condition_records"]
    }
    evidence_tasks = {
        row["scan_task_id"]: row
        for row in work_order["scan_tasks"]
        if row["counts_toward_evidence_minimum"] is True
    }
    if len(source_rows) != len(evidence_tasks):
        raise ThreeStockCaptureReceiptError("evidence scan-task count drift")

    observed_tasks: set[str] = set()
    for source, compiled in zip(source_rows, manifest_rows, strict=True):
        task_id = source.get("row_id")
        task = evidence_tasks.get(task_id)
        if task is None or task_id in observed_tasks:
            raise ThreeStockCaptureReceiptError("unknown or duplicate scan task")
        observed_tasks.add(task_id)
        exposure = exposure_by_id[task["exposure_slot_id"]]
        condition = condition_by_id[exposure["common_condition_slot_id"]]
        receipt_condition = receipt_condition_by_id[condition["condition_slot_id"]]
        expected = {
            "film_frame_id": exposure["exposure_slot_id"],
            "stock_id": exposure["stock_id"],
            "role": exposure["role"],
            "scene_id": condition["scene_id"],
            "roll_id": exposure["roll_slot_id"],
            "process_session_id": exposure["process_session_slot_id"],
            "process_type": exposure["process_type"],
            "interpretation_id": exposure["interpretation_id"],
            "scanner_device_id": task["scanner_device_slot_id"],
            "scanner_session_id": task["scanner_session_slot_id"],
            "camera_system_id": receipt_condition["camera_system_id"],
        }
        if any(source.get(key) != value for key, value in expected.items()):
            raise ThreeStockCaptureReceiptError(
                f"ledger/work-order identity drift: {task_id}"
            )
        if compiled["digital_reference_sha256"] != condition["stimulus_sha256"]:
            raise ThreeStockCaptureReceiptError(
                f"ledger stimulus identity drift: {task_id}"
            )
        condition_path = Path(str(source["capture_condition_record_path"]))
        if (
            condition_path.is_absolute()
            or ".." in condition_path.parts
            or not condition_path.parts
            or condition_path.parts[0].casefold() != "data"
        ):
            raise ThreeStockCaptureReceiptError("invalid capture-condition path")
        _, observed_condition = _read_object(root.joinpath(*condition_path.parts))
        if observed_condition != receipt_condition:
            raise ThreeStockCaptureReceiptError(
                f"capture-condition receipt drift: {task_id}"
            )

    if observed_tasks != set(evidence_tasks):
        raise ThreeStockCaptureReceiptError("evidence scan-task coverage drift")
    manifest_raw = _canonical(manifest)
    core = {
        "schema": BINDING_REPORT_SCHEMA,
        "experiment_id": "SF3.A0N-LEDGER",
        "receipt_contract_sha256": receipt_report["contract_sha256"],
        "work_order_stable_evidence_id": receipt_report[
            "work_order_stable_evidence_id"
        ],
        "receipt_packet_sha256": _sha256(packet_raw),
        "receipt_stable_evidence_id": receipt_report["stable_evidence_id"],
        "acquisition_contract_sha256": _sha256(acquisition_contract_raw),
        "acquisition_ledger_sha256": _sha256(ledger_raw),
        "compiled_manifest_sha256": _sha256(manifest_raw),
        "evidence_scan_tasks": len(evidence_tasks),
        "diagnostic_scan_tasks_excluded": len(work_order["scan_tasks"])
        - len(evidence_tasks),
        "automatic_pass": True,
        "decision": "READY_TO_RUN_SF3_A0_ADMISSION_WITH_RECEIPTS_BOUND",
        "pixel_reads": 0,
        "operator_fits": 0,
        "film_target_scores": 0,
        "claim_ceiling": (
            "Exact SF3.A0L/A0N receipt-to-ledger identity binding before SF3.A0 "
            "admission. Passing is not evidence that scans decode or align, is not "
            "a stock response, and creates no fitting, calibration or product claim."
        ),
    }
    return {
        **core,
        "compiled_manifest": manifest,
        "stable_evidence_id": _sha256(_canonical(core)),
    }


__all__ = [
    "BINDING_REPORT_SCHEMA",
    "ThreeStockCaptureReceiptError",
    "build_ledger_template",
    "build_receipt_template",
    "evaluate_ledger_binding",
    "evaluate_receipts",
]
