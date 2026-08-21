from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.real_film.three_stock_capture_work_order import (
    ThreeStockCaptureWorkOrderError,
    build_work_order,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/sf3_a0l_three_stock_capture_work_order_v1.json"


def test_builds_exact_three_stock_capture_and_scan_slots() -> None:
    first = build_work_order(CONTRACT, root=ROOT)
    second = build_work_order(CONTRACT, root=ROOT)
    assert first == second
    assert first["automatic_pass"] is True
    assert first["operator_fits"] == first["film_target_scores"] == 0
    assert first["counts"] == {
        "common_condition_records_to_fill": 18,
        "scene_exposures": 60,
        "diagnostic_exposures": 27,
        "evidence_scan_tasks": 108,
        "diagnostic_scan_tasks": 45,
        "total_scan_tasks": 153,
    }
    assert len(first["exposure_rows"]) == 87
    assert len({row["exposure_slot_id"] for row in first["exposure_rows"]}) == 87
    assert len({row["scan_task_id"] for row in first["scan_tasks"]}) == 153


def test_common_scene_condition_is_shared_but_exposure_receipts_are_per_frame() -> None:
    report = build_work_order(CONTRACT, root=ROOT)
    scene = "canon_eos_kiss_f"
    rows = [
        row
        for row in report["exposure_rows"]
        if row["kind"] == "scene" and row["common_condition_slot_id"].endswith(scene)
    ]
    assert len(rows) == 6
    assert len({row["common_condition_slot_id"] for row in rows}) == 1
    assert len({row["exposure_slot_id"] for row in rows}) == 6
    assert {row["nominal_iso"] for row in rows} == {50, 100, 400}
    assert report["exposure_policy"]["same_shutter_across_stocks_required"] is False
    assert "shutter_seconds" in rows[0]["exposure_receipt_required_fields"]


def test_role_slot_namespaces_are_disjoint() -> None:
    report = build_work_order(CONTRACT, root=ROOT)
    development = {
        row["scanner_device_slot_id"]
        for row in report["scan_tasks"]
        if row["role"] == "development"
    }
    confirmation = {
        row["scanner_device_slot_id"]
        for row in report["scan_tasks"]
        if row["role"] == "confirmation"
    }
    assert development.isdisjoint(confirmation)
    assert len(development) == 2
    assert len(confirmation) == 1


def test_parent_hash_drift_fails_before_work_order(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["parents"]["stimulus_manifest"]["sha256"] = "0" * 64
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ThreeStockCaptureWorkOrderError, match="identity drift"):
        load_contract(path, root=ROOT)
