from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.real_film.three_stock_capture_receipts import (
    ThreeStockCaptureReceiptError,
    build_receipt_template,
    evaluate_receipts,
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
