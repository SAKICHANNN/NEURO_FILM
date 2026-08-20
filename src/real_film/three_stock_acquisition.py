"""Compile and validate the SF3.A0 controlled three-stock acquisition intake."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA = "neuro-film.sf3-a0-three-stock-controlled-acquisition-contract.v1"
MANIFEST_SCHEMA = "neuro-film.sf3-a0-three-stock-controlled-acquisition-manifest.v1"
LEDGER_SCHEMA = "neuro-film.sf3-a0-three-stock-controlled-acquisition-ledger.v1"
REPORT_SCHEMA = "neuro-film.sf3-a0-three-stock-controlled-acquisition-report.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_HASH_PATH_FIELDS = {
    "digital_reference_sha256": "digital_reference_path",
    "capture_condition_sha256": "capture_condition_record_path",
    "process_recipe_sha256": "process_recipe_record_path",
    "scanner_profile_sha256": "scanner_profile_record_path",
    "scan_file_sha256": "scan_file_path",
    "scan_sample_sha256": "scan_sample_path",
    "alignment_evidence_sha256": "alignment_evidence_path",
    "rights_record_sha256": "rights_record_path",
}


class ThreeStockAcquisitionError(ValueError):
    """Raised when a contract or candidate manifest is structurally invalid."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _read_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ThreeStockAcquisitionError(f"expected JSON object: {path}")
    return raw, value


def _load_contract(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw, contract = _read_object(path)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ThreeStockAcquisitionError("unsupported SF3.A0 contract")
    if set(contract.get("roles", {})) != {"development", "confirmation"}:
        raise ThreeStockAcquisitionError("SF3.A0 role inventory drift")
    stocks = contract.get("required_stocks", {})
    if set(stocks) != {
        "fujifilm_velvia_50",
        "kodak_portra_400",
        "kodak_ektar_100",
    }:
        raise ThreeStockAcquisitionError("SF3.A0 stock inventory drift")
    fields = contract.get("required_row_fields", [])
    if len(fields) != len(set(fields)) or len(fields) < 20:
        raise ThreeStockAcquisitionError("SF3.A0 row field inventory drift")
    return raw, contract


def compile_protocol(contract_path: Path) -> dict[str, Any]:
    """Compile the frozen intake requirements without claiming acquired data."""

    raw, contract = _load_contract(contract_path)
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": _sha256(raw),
        "required_stocks": list(contract["required_stocks"]),
        "minimum_rows_if_exactly_at_threshold": sum(
            role["minimum_common_scenes"]
            * role["minimum_rolls_per_stock"]
            * role["minimum_scans_per_film_frame"]
            * len(contract["required_stocks"])
            for role in contract["roles"].values()
        ),
        "manifest_present": False,
        "automatic_pass": False,
        "decision": "READY_FOR_CONTROLLED_ACQUISITION_NO_DATA",
        "operator_fit_authority": False,
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": _sha256(_canonical(core))}


def _bound_data_file(root: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ThreeStockAcquisitionError(f"SF3.A0 ledger {field} is invalid")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ThreeStockAcquisitionError(
            f"SF3.A0 ledger {field} must be repository-relative"
        )
    if not relative.parts or relative.parts[0].casefold() != "data":
        raise ThreeStockAcquisitionError(
            f"SF3.A0 ledger {field} must use the logical data root"
        )
    path = root.joinpath(*relative.parts)
    if not path.is_file():
        raise ThreeStockAcquisitionError(f"SF3.A0 ledger file is missing: {value}")
    return path


def compile_acquisition_ledger(
    contract_path: Path, ledger_path: Path, *, root: Path
) -> dict[str, Any]:
    """Hash one controlled-capture ledger into the strict SF3.A0 manifest.

    The output deliberately contains identities rather than machine-specific
    physical paths. Pixel decoding and alignment adjudication remain SF3.A1.
    """

    _, contract = _load_contract(contract_path)
    _, ledger = _read_object(ledger_path)
    if set(ledger) != {"schema", "rows"}:
        raise ThreeStockAcquisitionError("SF3.A0 acquisition ledger field drift")
    if ledger.get("schema") != LEDGER_SCHEMA:
        raise ThreeStockAcquisitionError("unsupported SF3.A0 acquisition ledger")
    ledger_rows = ledger.get("rows")
    if not isinstance(ledger_rows, list) or not ledger_rows:
        raise ThreeStockAcquisitionError("SF3.A0 acquisition ledger rows are empty")
    manifest_fields = set(contract["required_row_fields"])
    ledger_fields = (manifest_fields - set(_HASH_PATH_FIELDS)) | set(
        _HASH_PATH_FIELDS.values()
    )
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(ledger_rows):
        if not isinstance(source, dict) or set(source) != ledger_fields:
            raise ThreeStockAcquisitionError(
                f"SF3.A0 ledger row {index} field drift"
            )
        row = {field: source[field] for field in manifest_fields - set(_HASH_PATH_FIELDS)}
        for hash_field, path_field in _HASH_PATH_FIELDS.items():
            path = _bound_data_file(root, source[path_field], field=path_field)
            row[hash_field] = _sha256_file(path)
        rows.append(row)
    _row_structure(contract, rows)
    return {"schema": MANIFEST_SCHEMA, "rows": rows}


def _row_structure(contract: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    required = set(contract["required_row_fields"])
    valid_stocks = set(contract["required_stocks"])
    valid_roles = set(contract["roles"])
    ids: set[str] = set()
    scan_hashes: set[str] = set()
    hash_fields = {
        "digital_reference_sha256",
        "capture_condition_sha256",
        "process_recipe_sha256",
        "scanner_profile_sha256",
        "scan_file_sha256",
        "scan_sample_sha256",
        "alignment_evidence_sha256",
        "rights_record_sha256",
    }
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != required:
            raise ThreeStockAcquisitionError(f"SF3.A0 row {index} field drift")
        if row["stock_id"] not in valid_stocks or row["role"] not in valid_roles:
            raise ThreeStockAcquisitionError(f"SF3.A0 row {index} class drift")
        if row["row_id"] in ids or row["scan_file_sha256"] in scan_hashes:
            raise ThreeStockAcquisitionError(f"SF3.A0 row {index} duplicate identity")
        ids.add(row["row_id"])
        scan_hashes.add(row["scan_file_sha256"])
        if any(not _SHA256.fullmatch(str(row[field])) for field in hash_fields):
            raise ThreeStockAcquisitionError(f"SF3.A0 row {index} hash drift")


def evaluate_manifest(contract_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Evaluate one candidate manifest without opening or decoding its pixel files."""

    contract_raw, contract = _load_contract(contract_path)
    manifest_raw, manifest = _read_object(manifest_path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ThreeStockAcquisitionError("unsupported SF3.A0 manifest")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ThreeStockAcquisitionError("SF3.A0 candidate rows are empty")
    _row_structure(contract, rows)

    required_stocks = set(contract["required_stocks"])
    rights = contract["rights"]
    observed_stocks = {row["stock_id"] for row in rows}
    rights_complete = all(
        row["rights_scope"] == rights["required_scope"]
        and row["rights_allow_internal_training"] is True
        and row["rights_allow_commercial_derivatives"] is True
        and row["rights_allow_released_weights"] is True
        for row in rows
    )
    process_interpretation_exact = all(
        row["process_type"]
        == contract["required_stocks"][row["stock_id"]]["process_type"]
        and row["interpretation_id"]
        == contract["required_stocks"][row["stock_id"]]["interpretation_id"]
        for row in rows
    )

    by_stock_role: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stock_role[(row["stock_id"], row["role"])].append(row)

    role_support = True
    scene_sets: dict[tuple[str, str], set[str]] = {}
    coverage: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for stock_id in contract["required_stocks"]:
        for role_name, requirements in contract["roles"].items():
            selected = by_stock_role[(stock_id, role_name)]
            scenes = {row["scene_id"] for row in selected}
            rolls = {row["roll_id"] for row in selected}
            processes = {row["process_session_id"] for row in selected}
            labs = {row["lab_id"] for row in selected}
            scanners = {row["scanner_session_id"] for row in selected}
            scanner_devices = {row["scanner_device_id"] for row in selected}
            scene_sets[(stock_id, role_name)] = scenes
            frames: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in selected:
                frames[row["film_frame_id"]].append(row)
            frame_identity_support = all(
                len(
                    {
                        (
                            row["stock_id"],
                            row["role"],
                            row["scene_id"],
                            row["scene_content_group"],
                            row["roll_id"],
                            row["capture_session_id"],
                            row["source_owner_id"],
                            row["camera_system_id"],
                            row["process_session_id"],
                            row["digital_reference_sha256"],
                            row["capture_condition_sha256"],
                            row["interpretation_id"],
                        )
                        for row in frame_rows
                    }
                )
                == 1
                for frame_rows in frames.values()
            ) and bool(frames)
            frame_scan_support = all(
                len({row["scanner_session_id"] for row in frame_rows})
                >= requirements["minimum_scans_per_film_frame"]
                for frame_rows in frames.values()
            ) and bool(frames)
            roll_scene_support = all(
                {row["scene_id"] for row in selected if row["roll_id"] == roll_id}
                == scenes
                for roll_id in rolls
            ) and bool(rolls)
            roll_identity_support = all(
                len(
                    {
                        (
                            row["process_session_id"],
                            row["lab_id"],
                            row["process_recipe_sha256"],
                        )
                        for row in selected
                        if row["roll_id"] == roll_id
                    }
                )
                == 1
                for roll_id in rolls
            ) and bool(rolls)
            scanner_identity_support = all(
                len(
                    {
                        (
                            row["scanner_device_id"],
                            row["scanner_profile_sha256"],
                        )
                        for row in selected
                        if row["scanner_session_id"] == scanner_id
                    }
                )
                == 1
                for scanner_id in scanners
            ) and bool(scanners)
            role_support &= (
                len(scenes) >= requirements["minimum_common_scenes"]
                and len(rolls) >= requirements["minimum_rolls_per_stock"]
                and len(processes)
                >= requirements["minimum_process_sessions_per_stock"]
                and len(labs) >= requirements["minimum_labs_per_stock"]
                and len(scanners)
                >= requirements["minimum_scanner_sessions_per_stock"]
                and len(scanner_devices)
                >= requirements["minimum_scanner_devices_per_stock"]
                and frame_identity_support
                and frame_scan_support
                and roll_scene_support
                and roll_identity_support
                and scanner_identity_support
            )
            coverage[stock_id][role_name] = {
                "scenes": len(scenes),
                "rolls": len(rolls),
                "process_sessions": len(processes),
                "labs": len(labs),
                "scanner_sessions": len(scanners),
                "scanner_devices": len(scanner_devices),
                "film_frames": len(frames),
            }

    same_scene_controls = True
    for role_name in contract["roles"]:
        role_scene_sets = {
            frozenset(scene_sets[(stock_id, role_name)])
            for stock_id in contract["required_stocks"]
        }
        same_scene_controls &= len(role_scene_sets) == 1
        role_scenes = next(iter(role_scene_sets), frozenset())
        for scene_id in role_scenes:
            selected = [
                row
                for row in rows
                if row["role"] == role_name and row["scene_id"] == scene_id
            ]
            same_scene_controls &= (
                len({row["digital_reference_sha256"] for row in selected}) == 1
                and len({row["capture_condition_sha256"] for row in selected}) == 1
                and len({row["source_owner_id"] for row in selected}) == 1
                and len({row["scene_content_group"] for row in selected}) == 1
                and len({row["camera_system_id"] for row in selected}) == 1
            )

    holdout_fields = (
        "scene_id",
        "roll_id",
        "process_session_id",
        "lab_id",
        "scanner_session_id",
        "scanner_device_id",
        "capture_session_id",
        "scene_content_group",
    )
    cross_role_holdout = all(
        not (
            {row[field] for row in rows if row["role"] == "development"}
            & {row[field] for row in rows if row["role"] == "confirmation"}
        )
        for field in holdout_fields
    )
    gates = {
        "stock_inventory_exact": observed_stocks == required_stocks,
        "rights_complete": rights_complete,
        "process_interpretation_exact": process_interpretation_exact,
        "role_support": role_support,
        "same_scene_controls": same_scene_controls,
        "cross_role_holdout": cross_role_holdout,
    }
    automatic_pass = all(gates.values())
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": _sha256(contract_raw),
        "manifest_sha256": _sha256(manifest_raw),
        "manifest_present": True,
        "row_count": len(rows),
        "coverage": coverage,
        "gates": gates,
        "automatic_pass": automatic_pass,
        "decision": contract[
            "decision_if_pass" if automatic_pass else "decision_if_fail"
        ],
        "pixel_reads": 0,
        "operator_fits": 0,
        "operator_fit_authority": False,
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": _sha256(_canonical(core))}


__all__ = [
    "CONTRACT_SCHEMA",
    "LEDGER_SCHEMA",
    "MANIFEST_SCHEMA",
    "ThreeStockAcquisitionError",
    "compile_acquisition_ledger",
    "compile_protocol",
    "evaluate_manifest",
]
