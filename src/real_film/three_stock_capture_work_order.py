"""Compile the SF3.A0K stimulus pack into a physical capture work order."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA = "neuro-film.sf3-a0l-three-stock-capture-work-order-contract.v1"
WORK_ORDER_SCHEMA = "neuro-film.sf3-a0l-three-stock-capture-work-order.v1"


class ThreeStockCaptureWorkOrderError(ValueError):
    """Raised when a frozen capture-work-order binding is invalid."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_file(root: Path, binding: dict[str, Any], *, label: str) -> Path:
    value = binding.get("path")
    if not isinstance(value, str) or not value:
        raise ThreeStockCaptureWorkOrderError(f"invalid {label} path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ThreeStockCaptureWorkOrderError(f"{label} must be repository-relative")
    path = root.joinpath(*relative.parts)
    if not path.is_file() or _sha256_file(path) != binding.get("sha256"):
        raise ThreeStockCaptureWorkOrderError(f"{label} identity drift")
    return path


def load_contract(path: Path, *, root: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    contract = json.loads(raw)
    if not isinstance(contract, dict) or contract.get("schema") != CONTRACT_SCHEMA:
        raise ThreeStockCaptureWorkOrderError("unsupported SF3.A0L contract")
    if contract.get("status") != "FROZEN_BEFORE_PHYSICAL_CAPTURE":
        raise ThreeStockCaptureWorkOrderError("SF3.A0L contract is not frozen")
    _repo_file(
        root, contract["parents"]["acquisition_contract"], label="acquisition contract"
    )
    _repo_file(
        root, contract["parents"]["stimulus_manifest"], label="stimulus manifest"
    )
    stock_ids = [row.get("stock_id") for row in contract.get("stocks", [])]
    if stock_ids != [
        "fujifilm_velvia_50",
        "kodak_portra_400",
        "kodak_ektar_100",
    ]:
        raise ThreeStockCaptureWorkOrderError("SF3.A0L stock order drift")
    if [row.get("nominal_iso") for row in contract["stocks"]] != [50, 400, 100]:
        raise ThreeStockCaptureWorkOrderError("SF3.A0L nominal ISO drift")
    exposure = contract.get("exposure_policy", {})
    if (
        exposure.get("film_ei_must_equal_nominal_iso") is not True
        or exposure.get("same_shutter_across_stocks_required") is not False
        or exposure.get("exposure_receipt_required_per_film_frame") is not True
    ):
        raise ThreeStockCaptureWorkOrderError("SF3.A0L exposure policy drift")
    return raw, contract


def _load_parent(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ThreeStockCaptureWorkOrderError(f"expected JSON object: {path}")
    return value


def build_work_order(contract_path: Path, *, root: Path) -> dict[str, Any]:
    """Build a deterministic work order without reading or inventing film data."""

    contract_raw, contract = load_contract(contract_path, root=root)
    acquisition = _load_parent(
        _repo_file(
            root,
            contract["parents"]["acquisition_contract"],
            label="acquisition contract",
        )
    )
    stimulus = _load_parent(
        _repo_file(
            root,
            contract["parents"]["stimulus_manifest"],
            label="stimulus manifest",
        )
    )
    required_stocks = list(acquisition.get("required_stocks", {}))
    stock_ids = [row["stock_id"] for row in contract["stocks"]]
    if required_stocks != stock_ids or stimulus.get("required_stocks") != stock_ids:
        raise ThreeStockCaptureWorkOrderError("parent stock inventory drift")
    stock_by_id = {row["stock_id"]: row for row in contract["stocks"]}
    for stock_id, stock in stock_by_id.items():
        expected = acquisition["required_stocks"][stock_id]
        if (
            stock["process_type"] != expected["process_type"]
            or stock["interpretation_id"] != expected["interpretation_id"]
        ):
            raise ThreeStockCaptureWorkOrderError("stock process policy drift")

    scenes_by_role: dict[str, list[dict[str, Any]]] = {
        "development": [],
        "confirmation": [],
    }
    for row in stimulus.get("scene_rows", []):
        if not isinstance(row, dict) or row.get("role") not in scenes_by_role:
            raise ThreeStockCaptureWorkOrderError("stimulus scene inventory drift")
        scenes_by_role[row["role"]].append(row)
    for role, rows in scenes_by_role.items():
        rows.sort(key=lambda row: int(row["role_order"]))
        expected = int(acquisition["roles"][role]["minimum_common_scenes"])
        if len(rows) != expected or len({row["scene_id"] for row in rows}) != expected:
            raise ThreeStockCaptureWorkOrderError("stimulus role scene count drift")

    diagnostics = stimulus.get("diagnostic_rows", [])
    if len(diagnostics) != 3 or any(
        row.get("counts_toward_scene_minima") is not False for row in diagnostics
    ):
        raise ThreeStockCaptureWorkOrderError("diagnostic inventory drift")

    common_conditions: dict[str, dict[str, Any]] = {}
    exposure_rows: list[dict[str, Any]] = []
    scan_tasks: list[dict[str, Any]] = []
    for role in ("development", "confirmation"):
        role_slots = contract["role_slots"][role]
        rolls = int(role_slots["rolls_per_stock"])
        scanners = int(role_slots["scanner_devices_per_frame"])
        if rolls != int(acquisition["roles"][role]["minimum_rolls_per_stock"]):
            raise ThreeStockCaptureWorkOrderError("roll slot count drift")
        if scanners != int(acquisition["roles"][role]["minimum_scans_per_film_frame"]):
            raise ThreeStockCaptureWorkOrderError("scanner slot count drift")
        for scene in scenes_by_role[role]:
            condition_id = f"{role}:scene:{scene['scene_id']}"
            common_conditions[condition_id] = {
                "condition_slot_id": condition_id,
                "role": role,
                "scene_id": scene["scene_id"],
                "stimulus_sha256": scene["stimulus_sha256"],
                "stimulus_relative_path": scene["stimulus_relative_path"],
                "required_fields": contract["exposure_policy"][
                    "common_condition_fields"
                ],
            }
        for diagnostic in diagnostics:
            condition_id = f"{role}:diagnostic:{diagnostic['diagnostic_id']}"
            common_conditions[condition_id] = {
                "condition_slot_id": condition_id,
                "role": role,
                "diagnostic_id": diagnostic["diagnostic_id"],
                "stimulus_sha256": diagnostic["sha256"],
                "stimulus_relative_path": diagnostic["relative_path"],
                "required_fields": contract["exposure_policy"][
                    "common_condition_fields"
                ],
            }
        for stock_id in stock_ids:
            stock = stock_by_id[stock_id]
            for roll_index in range(1, rolls + 1):
                roll_slot = f"{role}:{stock_id}:roll:{roll_index:02d}"
                process_slot = f"{role}:{stock_id}:process:{roll_index:02d}"
                for kind, rows in (
                    ("scene", scenes_by_role[role]),
                    ("diagnostic", diagnostics),
                ):
                    for row in rows:
                        item_id = (
                            row["scene_id"] if kind == "scene" else row["diagnostic_id"]
                        )
                        condition_id = f"{role}:{kind}:{item_id}"
                        exposure_slot = f"{roll_slot}:{kind}:{item_id}"
                        exposure_rows.append(
                            {
                                "exposure_slot_id": exposure_slot,
                                "kind": kind,
                                "role": role,
                                "stock_id": stock_id,
                                "nominal_iso": stock["nominal_iso"],
                                "process_type": stock["process_type"],
                                "interpretation_id": stock["interpretation_id"],
                                "roll_slot_id": roll_slot,
                                "process_session_slot_id": process_slot,
                                "common_condition_slot_id": condition_id,
                                "exposure_receipt_required_fields": contract[
                                    "exposure_policy"
                                ]["stock_exposure_receipt_fields"],
                            }
                        )
                        for scanner_index in range(1, scanners + 1):
                            scan_tasks.append(
                                {
                                    "scan_task_id": f"{exposure_slot}:scan:{scanner_index:02d}",
                                    "exposure_slot_id": exposure_slot,
                                    "kind": kind,
                                    "role": role,
                                    "stock_id": stock_id,
                                    "scanner_device_slot_id": f"{role}:scanner-device:{scanner_index:02d}",
                                    "scanner_session_slot_id": f"{role}:scanner-session:{scanner_index:02d}",
                                    "counts_toward_evidence_minimum": kind == "scene",
                                }
                            )

    scene_exposures = sum(row["kind"] == "scene" for row in exposure_rows)
    diagnostic_exposures = len(exposure_rows) - scene_exposures
    evidence_scans = sum(row["counts_toward_evidence_minimum"] for row in scan_tasks)
    diagnostic_scans = len(scan_tasks) - evidence_scans
    gates = contract["gates"]
    counts = {
        "common_condition_records_to_fill": len(common_conditions),
        "scene_exposures": scene_exposures,
        "diagnostic_exposures": diagnostic_exposures,
        "evidence_scan_tasks": evidence_scans,
        "diagnostic_scan_tasks": diagnostic_scans,
        "total_scan_tasks": len(scan_tasks),
    }
    if (
        scene_exposures != gates["expected_scene_exposures"]
        or diagnostic_exposures != gates["expected_diagnostic_exposures"]
        or evidence_scans != gates["expected_evidence_scan_rows"]
        or diagnostic_scans != gates["expected_diagnostic_scan_rows"]
        or len(scan_tasks) != gates["expected_total_scan_tasks"]
    ):
        raise ThreeStockCaptureWorkOrderError("derived work-order count drift")
    core = {
        "schema": WORK_ORDER_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": _sha256(contract_raw),
        "acquisition_contract_sha256": contract["parents"]["acquisition_contract"][
            "sha256"
        ],
        "stimulus_manifest_sha256": contract["parents"]["stimulus_manifest"]["sha256"],
        "stocks": contract["stocks"],
        "exposure_policy": contract["exposure_policy"],
        "counts": counts,
        "common_condition_records": list(common_conditions.values()),
        "exposure_rows": exposure_rows,
        "scan_tasks": scan_tasks,
        "operator_fits": 0,
        "film_target_scores": 0,
        "automatic_pass": True,
        "decision": contract["decision_if_pass"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": _sha256(_canonical(core))}


__all__ = [
    "CONTRACT_SCHEMA",
    "WORK_ORDER_SCHEMA",
    "ThreeStockCaptureWorkOrderError",
    "build_work_order",
    "load_contract",
]
