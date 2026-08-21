"""Preflight the fixed P-backed SF3 colour-baseline scan tier."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

SCHEMA = "neuro-film.sf3-a0p-three-stock-scan-storage-preflight-contract.v1"
REPORT_SCHEMA = "neuro-film.sf3-a0p-three-stock-scan-storage-preflight-report.v1"


class ThreeStockScanStoragePreflightError(ValueError):
    """Raised when the fixed storage preflight contract drifts."""


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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bound_file(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    relative = Path(str(binding.get("path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ThreeStockScanStoragePreflightError("parent path must be repository-relative")
    path = root.joinpath(*relative.parts)
    if not path.is_file() or _sha256_file(path) != binding.get("sha256"):
        raise ThreeStockScanStoragePreflightError("parent identity drift")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ThreeStockScanStoragePreflightError("parent must be a JSON object")
    return path, value


def load_contract(path: Path, *, root: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    contract = json.loads(raw)
    if (
        not isinstance(contract, dict)
        or contract.get("schema") != SCHEMA
        or contract.get("experiment_id") != "SF3.A0P"
        or contract.get("status") != "FROZEN_BEFORE_PHYSICAL_SCAN_WRITE"
    ):
        raise ThreeStockScanStoragePreflightError("unsupported SF3.A0P contract")
    _, work_order = _bound_file(root, contract["parents"]["work_order"])
    _, stimulus = _bound_file(root, contract["parents"]["stimulus_manifest"])
    if work_order.get("decision") != contract["parents"]["work_order"]["required_decision"]:
        raise ThreeStockScanStoragePreflightError("work-order decision drift")
    if not stimulus.get("scene_rows"):
        raise ThreeStockScanStoragePreflightError("stimulus scene inventory is empty")
    profile = contract.get("scan_profile", {})
    if (
        profile.get("width") != 3000
        or profile.get("height") != 2000
        or profile.get("channels") != 3
        or profile.get("integer_bits_per_channel") != 16
        or profile.get("lossless_required") is not True
    ):
        raise ThreeStockScanStoragePreflightError("scan profile drift")
    return raw, contract


def evaluate(path: Path, *, root: Path) -> dict[str, Any]:
    """Evaluate current capacity without creating the physical-capture root."""

    raw, contract = load_contract(path, root=root)
    _, work_order = _bound_file(root, contract["parents"]["work_order"])
    stimulus_path, stimulus = _bound_file(
        root, contract["parents"]["stimulus_manifest"]
    )
    counts = work_order.get("counts", {})
    count_exact = all(
        int(counts.get(key, -1)) == int(value)
        for key, value in contract["required_counts"].items()
    )
    max_stimulus_pixels = max(
        int(row["width"]) * int(row["height"]) for row in stimulus["scene_rows"]
    )
    profile = contract["scan_profile"]
    scan_pixels = int(profile["width"]) * int(profile["height"])
    bytes_per_sample = (int(profile["integer_bits_per_channel"]) + 7) // 8
    payload_per_file = scan_pixels * int(profile["channels"]) * bytes_per_sample
    worst_file_bytes = payload_per_file + int(
        profile["maximum_container_overhead_bytes_per_file"]
    )
    worst_plan_bytes = worst_file_bytes * int(
        contract["required_counts"]["total_scan_tasks"]
    )
    storage = contract["storage"]
    logical = root.joinpath(*Path(storage["logical_root"]).parts)
    resolved_data = (root / "data").resolve()
    resolved_drive = resolved_data.drive.upper()
    available = int(shutil.disk_usage(resolved_data).free)
    remaining = available - worst_plan_bytes
    ratio = scan_pixels / max_stimulus_pixels
    gates = {
        "work_order_counts_exact": count_exact,
        "logical_output_root_is_create_only_absent": not logical.exists(),
        "resolved_storage_drive_exact": resolved_drive
        == str(storage["required_resolved_drive"]).upper(),
        "scan_resolution_exceeds_stimulus_ratio": ratio
        >= float(profile["minimum_scan_to_stimulus_pixel_ratio"]),
        "worst_case_plan_preserves_free_space": remaining
        >= int(storage["minimum_free_bytes_after_worst_case_plan"]),
    }
    automatic_pass = all(gates.values())
    scientific = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": _sha256(raw),
        "work_order_sha256": contract["parents"]["work_order"]["sha256"],
        "stimulus_manifest_sha256": _sha256_file(stimulus_path),
        "scan_profile": profile,
        "scan_tasks": int(contract["required_counts"]["total_scan_tasks"]),
        "maximum_stimulus_pixels": max_stimulus_pixels,
        "scan_pixels": scan_pixels,
        "scan_to_stimulus_pixel_ratio": ratio,
        "worst_case_file_bytes": worst_file_bytes,
        "worst_case_plan_bytes": worst_plan_bytes,
        "required_free_bytes_after_plan": int(
            storage["minimum_free_bytes_after_worst_case_plan"]
        ),
        "gates": gates,
        "automatic_pass": automatic_pass,
        "decision": contract[
            "decision_if_pass" if automatic_pass else "decision_if_fail"
        ],
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable = _sha256(_canonical(scientific))
    return {
        **scientific,
        "observed_available_bytes": available,
        "projected_remaining_bytes": remaining,
        "resolved_data_root": str(resolved_data),
        "stable_evidence_id": stable,
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "ThreeStockScanStoragePreflightError",
    "evaluate",
    "load_contract",
]
