from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import src.real_film.three_stock_capture_receipts as receipts_module
from src.real_film.three_stock_capture_receipts import (
    ThreeStockCaptureReceiptError,
    build_receipt_template,
    build_single_stock_ledger_template,
    build_single_stock_receipt_template,
    evaluate_ledger_binding,
    evaluate_receipts,
    evaluate_single_stock_ledger_binding,
    evaluate_single_stock_receipts,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/sf3_a0n_three_stock_capture_receipts_v1.json"
WORK_ORDER = (
    ROOT / "outputs/eval/sf3_a0l_three_stock_capture_work_order_v1/run_a/report.json"
)


def _packet() -> dict:
    work = json.loads(WORK_ORDER.read_text(encoding="utf-8"))
    common = []
    for expected in work["common_condition_records"]:
        common.append(
            {
                "condition_slot_id": expected["condition_slot_id"],
                "stimulus_sha256": expected["stimulus_sha256"],
                "display_device_id": "display-1",
                "display_profile_sha256": "a" * 64,
                "display_luminance_cd_m2": 120.0,
                "ambient_illuminance_lux": 5.0,
                "camera_system_id": "camera-1",
                "lens_id": "lens-1",
                "aperture_f_number": 8.0,
                "focus_distance_m": 2.0,
                "framing_id": "fixed-frame-1",
            }
        )
    exposures = []
    for expected in work["exposure_rows"]:
        exposures.append(
            {
                "exposure_slot_id": expected["exposure_slot_id"],
                "stock_id": expected["stock_id"],
                "film_ei": expected["nominal_iso"],
                "nominal_iso": expected["nominal_iso"],
                "shutter_seconds": 1.0 / expected["nominal_iso"],
                "meter_reading_ev100": 10.0,
                "exposure_compensation_ev": 0.0,
                "meter_id": "meter-1",
                "meter_calibration_sha256": "b" * 64,
            }
        )
    return {
        "schema": "neuro-film.sf3-a0n-three-stock-capture-receipt-packet.v1",
        "work_order_stable_evidence_id": work["stable_evidence_id"],
        "common_condition_records": common,
        "exposure_receipts": exposures,
    }


def _write(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "packet.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_complete_receipts_pass_without_fitting(tmp_path: Path) -> None:
    report = evaluate_receipts(CONTRACT, _write(tmp_path, _packet()), root=ROOT)
    assert report["automatic_pass"] is True
    assert report["common_condition_records"] == 18
    assert report["exposure_receipts"] == 87
    assert report["operator_fits"] == report["film_target_scores"] == 0


def test_template_has_exact_slots_and_only_prefills_bound_identities() -> None:
    template = build_receipt_template(CONTRACT, root=ROOT)
    assert len(template["common_condition_records"]) == 18
    assert len(template["exposure_receipts"]) == 87
    first = template["exposure_receipts"][0]
    assert first["stock_id"] == "fujifilm_velvia_50"
    assert first["film_ei"] == first["nominal_iso"] == 50
    assert first["shutter_seconds"] is None
    assert template == build_receipt_template(CONTRACT, root=ROOT)


def test_single_stock_receipts_and_ledger_can_advance_independently(
    tmp_path: Path,
) -> None:
    stock = "kodak_ektar_100"
    packet = _packet()
    packet["schema"] = "neuro-film.sf3-a0n-single-stock-capture-receipt-packet.v1"
    packet["stock"] = stock
    packet["exposure_receipts"] = [
        row for row in packet["exposure_receipts"] if row["stock_id"] == stock
    ]
    work = json.loads(WORK_ORDER.read_text(encoding="utf-8"))
    exposure_ids = {row["exposure_slot_id"] for row in packet["exposure_receipts"]}
    condition_ids = {
        row["common_condition_slot_id"]
        for row in work["exposure_rows"]
        if row["exposure_slot_id"] in exposure_ids
    }
    packet["common_condition_records"] = [
        row
        for row in packet["common_condition_records"]
        if row["condition_slot_id"] in condition_ids
    ]
    packet_path = _write(tmp_path, packet)
    report = evaluate_single_stock_receipts(
        CONTRACT, packet_path, root=ROOT, stock=stock
    )
    template = build_single_stock_receipt_template(CONTRACT, root=ROOT, stock=stock)
    ledger = build_single_stock_ledger_template(
        CONTRACT,
        packet_path,
        ROOT / "configs/sf3_a0_three_stock_controlled_acquisition_v1.json",
        root=ROOT,
        stock=stock,
    )
    assert report["automatic_pass"] is True
    assert report["stock"] == stock
    assert len(template["exposure_receipts"]) == len(packet["exposure_receipts"])
    assert len(template["common_condition_records"]) == len(
        packet["common_condition_records"]
    )
    assert len(ledger["rows"]) == 36
    assert {row["stock_id"] for row in ledger["rows"]} == {stock}


@pytest.mark.parametrize("mutation", ["missing", "wrong_ei", "nan", "bad_hash"])
def test_invalid_receipts_fail_closed(tmp_path: Path, mutation: str) -> None:
    packet = _packet()
    if mutation == "missing":
        packet["exposure_receipts"].pop()
    elif mutation == "wrong_ei":
        packet["exposure_receipts"][0]["film_ei"] = 400
    elif mutation == "nan":
        packet["common_condition_records"][0]["display_luminance_cd_m2"] = float("nan")
    else:
        packet["exposure_receipts"][0]["meter_calibration_sha256"] = "0"
    with pytest.raises(ThreeStockCaptureReceiptError):
        evaluate_receipts(CONTRACT, _write(tmp_path, packet), root=ROOT)


def _binding_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    root = tmp_path / "repo"
    (tmp_path / "acquisition.json").write_text("{}", encoding="utf-8")
    condition_path = root / "data" / "condition.json"
    condition_path.parent.mkdir(parents=True)
    condition = {
        "condition_slot_id": "development:scene:scene-1",
        "stimulus_sha256": "1" * 64,
        "display_device_id": "display-1",
        "display_profile_sha256": "2" * 64,
        "display_luminance_cd_m2": 120.0,
        "ambient_illuminance_lux": 5.0,
        "camera_system_id": "camera-1",
        "lens_id": "lens-1",
        "aperture_f_number": 8.0,
        "focus_distance_m": 2.0,
        "framing_id": "frame-1",
    }
    condition_path.write_text(json.dumps(condition), encoding="utf-8")
    packet = {
        "common_condition_records": [condition],
        "exposure_receipts": [],
    }
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    source = {
        "row_id": "development:fujifilm_velvia_50:roll:01:scene:scene-1:scan:01",
        "film_frame_id": "development:fujifilm_velvia_50:roll:01:scene:scene-1",
        "stock_id": "fujifilm_velvia_50",
        "role": "development",
        "scene_id": "scene-1",
        "roll_id": "development:fujifilm_velvia_50:roll:01",
        "process_session_id": "development:fujifilm_velvia_50:process:01",
        "process_type": "e6_reversal",
        "interpretation_id": "direct_slide_neutral_scan",
        "scanner_device_id": "development:scanner-device:01",
        "scanner_session_id": "development:scanner-session:01",
        "camera_system_id": "camera-1",
        "capture_condition_record_path": "data/condition.json",
    }
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps({"schema": "test-ledger", "rows": [source]}), encoding="utf-8"
    )
    work_order = {
        "exposure_rows": [
            {
                "exposure_slot_id": source["film_frame_id"],
                "stock_id": source["stock_id"],
                "role": source["role"],
                "roll_slot_id": source["roll_id"],
                "process_session_slot_id": source["process_session_id"],
                "process_type": source["process_type"],
                "interpretation_id": source["interpretation_id"],
                "common_condition_slot_id": condition["condition_slot_id"],
            }
        ],
        "common_condition_records": [
            {
                "condition_slot_id": condition["condition_slot_id"],
                "scene_id": source["scene_id"],
                "stimulus_sha256": condition["stimulus_sha256"],
            }
        ],
        "scan_tasks": [
            {
                "scan_task_id": source["row_id"],
                "exposure_slot_id": source["film_frame_id"],
                "scanner_device_slot_id": source["scanner_device_id"],
                "scanner_session_slot_id": source["scanner_session_id"],
                "counts_toward_evidence_minimum": True,
            },
            {
                "scan_task_id": "diagnostic-task",
                "counts_toward_evidence_minimum": False,
            },
        ],
    }
    manifest = {
        "schema": "test-manifest",
        "rows": [{"digital_reference_sha256": condition["stimulus_sha256"]}],
    }
    monkeypatch.setattr(
        receipts_module,
        "evaluate_receipts",
        lambda *_args, **_kwargs: {
            "contract_sha256": "3" * 64,
            "work_order_stable_evidence_id": "4" * 64,
            "stable_evidence_id": "5" * 64,
        },
    )
    monkeypatch.setattr(
        receipts_module,
        "_load_contract_and_work_order",
        lambda *_args, **_kwargs: (b"{}", {}, work_order),
    )
    monkeypatch.setattr(
        receipts_module,
        "compile_acquisition_ledger",
        lambda *_args, **_kwargs: manifest,
    )
    return root, packet_path, ledger_path, source


def test_receipt_packet_binds_exact_evidence_ledger_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, packet, ledger, _ = _binding_fixture(tmp_path, monkeypatch)
    report = evaluate_ledger_binding(
        tmp_path / "contract.json",
        packet,
        tmp_path / "acquisition.json",
        ledger,
        root=root,
    )
    assert report["automatic_pass"] is True
    assert report["evidence_scan_tasks"] == 1
    assert report["diagnostic_scan_tasks_excluded"] == 1
    assert report["pixel_reads"] == report["operator_fits"] == 0
    assert report["compiled_manifest"]["schema"] == "test-manifest"
    manifest_bytes = (
        json.dumps(report["compiled_manifest"], sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")
    assert (
        report["compiled_manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    )


def test_single_stock_receipt_packet_binds_only_its_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, packet, ledger, source = _binding_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        receipts_module,
        "evaluate_single_stock_receipts",
        lambda *_args, **_kwargs: {
            "contract_sha256": "3" * 64,
            "work_order_stable_evidence_id": "4" * 64,
            "stable_evidence_id": "5" * 64,
        },
    )
    report = evaluate_single_stock_ledger_binding(
        tmp_path / "contract.json",
        packet,
        tmp_path / "acquisition.json",
        ledger,
        root=root,
        stock=source["stock_id"],
    )
    assert report["automatic_pass"] is True
    assert report["stock"] == source["stock_id"]
    assert report["cross_stock_binding_evaluated"] is False
    assert report["evidence_scan_tasks"] == 1


def test_receipt_packet_rejects_ledger_slot_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, packet, ledger, _ = _binding_fixture(tmp_path, monkeypatch)
    value = json.loads(ledger.read_text(encoding="utf-8"))
    value["rows"][0]["film_frame_id"] = "foreign-frame"
    ledger.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ThreeStockCaptureReceiptError, match="identity drift"):
        evaluate_ledger_binding(
            tmp_path / "contract.json",
            packet,
            tmp_path / "acquisition.json",
            ledger,
            root=root,
        )
