from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.real_film.three_stock_acquisition import LEDGER_SCHEMA, evaluate_manifest
from src.real_film.three_stock_capture_receipts import (
    ThreeStockCaptureReceiptError,
    build_ledger_template,
    evaluate_ledger_binding,
    evaluate_receipts,
)

ROOT = Path(__file__).resolve().parents[1]
RECEIPT_CONTRACT = ROOT / "configs/sf3_a0n_three_stock_capture_receipts_v1.json"
ACQUISITION_CONTRACT = (
    ROOT / "configs/sf3_a0_three_stock_controlled_acquisition_v1.json"
)
WORK_ORDER = (
    ROOT / "outputs/eval/sf3_a0l_three_stock_capture_work_order_v1/run_a/report.json"
)
STIMULUS_ROOT = ROOT / "data/real_film/sf3_a0k_three_stock_display_stimulus_v1"


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path.as_posix()


def _repo_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _safe_name(value: str) -> str:
    return value.replace(":", "_")


def _build_packet(work: dict) -> dict:
    conditions = []
    for row in work["common_condition_records"]:
        conditions.append(
            {
                "condition_slot_id": row["condition_slot_id"],
                "stimulus_sha256": row["stimulus_sha256"],
                "display_device_id": f"display-{row['role']}",
                "display_profile_sha256": "a" * 64,
                "display_luminance_cd_m2": 120.0,
                "ambient_illuminance_lux": 5.0,
                "camera_system_id": f"camera-{row['role']}",
                "lens_id": f"lens-{row['role']}",
                "aperture_f_number": 8.0,
                "focus_distance_m": 2.0,
                "framing_id": f"frame-{row['role']}",
            }
        )
    exposures = []
    for row in work["exposure_rows"]:
        exposures.append(
            {
                "exposure_slot_id": row["exposure_slot_id"],
                "stock_id": row["stock_id"],
                "film_ei": row["nominal_iso"],
                "nominal_iso": row["nominal_iso"],
                "shutter_seconds": 1.0 / row["nominal_iso"],
                "meter_reading_ev100": 10.0,
                "exposure_compensation_ev": 0.0,
                "meter_id": "meter-1",
                "meter_calibration_sha256": "b" * 64,
            }
        )
    return {
        "schema": "neuro-film.sf3-a0n-three-stock-capture-receipt-packet.v1",
        "work_order_stable_evidence_id": work["stable_evidence_id"],
        "common_condition_records": conditions,
        "exposure_receipts": exposures,
    }


def test_real_a0l_a0n_to_a0_manifest_chain(tmp_path: Path) -> None:
    copied_work_order = tmp_path / WORK_ORDER.relative_to(ROOT)
    copied_work_order.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(WORK_ORDER, copied_work_order)
    work = json.loads(WORK_ORDER.read_text(encoding="utf-8"))
    packet = _build_packet(work)
    packet_path = tmp_path / "packet.json"
    _write_json(packet_path, packet)
    assert evaluate_receipts(RECEIPT_CONTRACT, packet_path, root=ROOT)["automatic_pass"]

    data_root = tmp_path / "data/sf3"
    condition_by_id = {
        row["condition_slot_id"]: row for row in packet["common_condition_records"]
    }
    expected_condition_by_id = {
        row["condition_slot_id"]: row for row in work["common_condition_records"]
    }
    exposure_by_id = {row["exposure_slot_id"]: row for row in work["exposure_rows"]}
    condition_paths: dict[str, Path] = {}
    digital_paths: dict[str, Path] = {}
    for condition_id, receipt in condition_by_id.items():
        condition_paths[condition_id] = (
            data_root / "conditions" / f"{_safe_name(condition_id)}.json"
        )
        _write_json(condition_paths[condition_id], receipt)
        expected = expected_condition_by_id[condition_id]
        destination = data_root / "stimuli" / expected["stimulus_relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(STIMULUS_ROOT / expected["stimulus_relative_path"], destination)
        digital_paths[condition_id] = destination

    rows = []
    for task in work["scan_tasks"]:
        if task["counts_toward_evidence_minimum"] is not True:
            continue
        exposure = exposure_by_id[task["exposure_slot_id"]]
        condition_id = exposure["common_condition_slot_id"]
        condition = expected_condition_by_id[condition_id]
        role = exposure["role"]
        process = (
            data_root
            / "process"
            / f"{_safe_name(exposure['process_session_slot_id'])}.json"
        )
        scanner = (
            data_root
            / "scanner"
            / f"{_safe_name(task['scanner_session_slot_id'])}.json"
        )
        rights = data_root / "rights" / f"{role}.json"
        safe_task_id = _safe_name(task["scan_task_id"])
        scan = data_root / "scan" / f"{safe_task_id}.tif"
        sample = data_root / "sample" / f"{safe_task_id}.tif"
        alignment = data_root / "alignment" / f"{safe_task_id}.json"
        for path, value in (
            (process, {"process": exposure["process_session_slot_id"]}),
            (scanner, {"scanner": task["scanner_session_slot_id"]}),
            (rights, {"owner": "project-owner", "role": role}),
            (alignment, {"row_id": task["scan_task_id"]}),
        ):
            if not path.exists():
                _write_json(path, value)
        scan.parent.mkdir(parents=True, exist_ok=True)
        sample.parent.mkdir(parents=True, exist_ok=True)
        scan.write_bytes(f"scan:{task['scan_task_id']}".encode("ascii"))
        sample.write_bytes(f"sample:{task['scan_task_id']}".encode("ascii"))
        rows.append(
            {
                "row_id": task["scan_task_id"],
                "stock_id": exposure["stock_id"],
                "role": role,
                "scene_id": condition["scene_id"],
                "scene_content_group": f"content:{role}:{condition['scene_id']}",
                "film_frame_id": exposure["exposure_slot_id"],
                "capture_session_id": f"capture:{role}",
                "source_owner_id": "project-owner",
                "camera_system_id": condition_by_id[condition_id]["camera_system_id"],
                "digital_reference_path": _repo_path(
                    tmp_path, digital_paths[condition_id]
                ),
                "capture_condition_record_path": _repo_path(
                    tmp_path, condition_paths[condition_id]
                ),
                "roll_id": exposure["roll_slot_id"],
                "process_session_id": exposure["process_session_slot_id"],
                "process_type": exposure["process_type"],
                "lab_id": f"lab:{role}",
                "process_recipe_record_path": _repo_path(tmp_path, process),
                "scanner_session_id": task["scanner_session_slot_id"],
                "scanner_device_id": task["scanner_device_slot_id"],
                "scanner_profile_record_path": _repo_path(tmp_path, scanner),
                "scan_file_path": _repo_path(tmp_path, scan),
                "scan_sample_path": _repo_path(tmp_path, sample),
                "alignment_evidence_path": _repo_path(tmp_path, alignment),
                "interpretation_id": exposure["interpretation_id"],
                "rights_record_path": _repo_path(tmp_path, rights),
                "rights_scope": "project_owned_internal_research_and_commercial_derivatives",
                "rights_allow_internal_training": True,
                "rights_allow_commercial_derivatives": True,
                "rights_allow_released_weights": True,
            }
        )

    ledger_path = tmp_path / "ledger.json"
    _write_json(ledger_path, {"schema": LEDGER_SCHEMA, "rows": rows})
    binding = evaluate_ledger_binding(
        RECEIPT_CONTRACT,
        packet_path,
        ACQUISITION_CONTRACT,
        ledger_path,
        root=tmp_path,
    )
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, binding["compiled_manifest"])
    admission = evaluate_manifest(ACQUISITION_CONTRACT, manifest_path)
    assert binding["automatic_pass"] is True
    assert binding["evidence_scan_tasks"] == admission["row_count"] == 108
    assert binding["diagnostic_scan_tasks_excluded"] == 45
    assert admission["automatic_pass"] is True


def test_verified_receipts_build_exact_evidence_ledger_skeleton(tmp_path: Path) -> None:
    work = json.loads(WORK_ORDER.read_text(encoding="utf-8"))
    packet_path = tmp_path / "packet.json"
    _write_json(packet_path, _build_packet(work))

    ledger = build_ledger_template(
        RECEIPT_CONTRACT,
        packet_path,
        ACQUISITION_CONTRACT,
        root=ROOT,
    )

    assert ledger["schema"] == LEDGER_SCHEMA
    assert len(ledger["rows"]) == 108
    assert len({row["row_id"] for row in ledger["rows"]}) == 108
    assert {row["stock_id"] for row in ledger["rows"]} == {
        "fujifilm_velvia_50",
        "kodak_portra_400",
        "kodak_ektar_100",
    }
    first = ledger["rows"][0]
    assert first["film_frame_id"] is not None
    assert first["camera_system_id"] is not None
    assert first["scanner_device_id"] is not None
    assert first["digital_reference_path"] is None
    assert first["scan_sample_path"] is None
    assert first["rights_scope"] is None

    drifted_contract = tmp_path / "acquisition.json"
    drifted = json.loads(ACQUISITION_CONTRACT.read_text(encoding="utf-8"))
    drifted["question"] = "drift"
    _write_json(drifted_contract, drifted)
    with pytest.raises(ThreeStockCaptureReceiptError, match="identity drift"):
        build_ledger_template(
            RECEIPT_CONTRACT,
            packet_path,
            drifted_contract,
            root=ROOT,
        )
