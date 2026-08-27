from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.update_sf3_a0n_capture_receipt_session import main as session_cli
from src.real_film.three_stock_capture_receipts import (
    ThreeStockCaptureReceiptError,
    build_receipt_template,
    evaluate_receipts,
)
from src.real_film.three_stock_capture_session import (
    CONDITION_KIND,
    EXPOSURE_KIND,
    capture_session_progress,
    update_capture_session_batch,
    update_capture_session_row,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/sf3_a0n_three_stock_capture_receipts_v1.json"


def _condition_values() -> dict[str, object]:
    return {
        "display_device_id": "display-v1",
        "display_profile_sha256": "a" * 64,
        "display_luminance_cd_m2": 120.0,
        "ambient_illuminance_lux": 5.0,
        "camera_system_id": "camera-v1",
        "lens_id": "lens-v1",
        "aperture_f_number": 8.0,
        "focus_distance_m": 2.0,
        "framing_id": "framing-v1",
    }


def _exposure_values() -> dict[str, object]:
    return {
        "shutter_seconds": 0.01,
        "meter_reading_ev100": 10.0,
        "exposure_compensation_ev": 0.0,
        "meter_id": "meter-v1",
        "meter_calibration_sha256": "b" * 64,
    }


def test_updates_are_copy_on_write_and_progress_in_frozen_order() -> None:
    packet = build_receipt_template(CONTRACT, root=ROOT)
    initial = capture_session_progress(CONTRACT, packet, root=ROOT)
    assert initial["conditions"] == {
        "total": 18,
        "filled": 0,
        "remaining": 18,
        "next_unfilled_id": packet["common_condition_records"][0]["condition_slot_id"],
    }
    condition_id = initial["conditions"]["next_unfilled_id"]
    after_condition = update_capture_session_row(
        CONTRACT,
        packet,
        root=ROOT,
        kind=CONDITION_KIND,
        slot_id=condition_id,
        values=_condition_values(),
    )
    assert packet["common_condition_records"][0]["display_device_id"] is None
    assert (
        capture_session_progress(CONTRACT, after_condition, root=ROOT)["conditions"][
            "filled"
        ]
        == 1
    )

    exposure_id = initial["exposures"]["next_unfilled_id"]
    after_exposure = update_capture_session_row(
        CONTRACT,
        after_condition,
        root=ROOT,
        kind=EXPOSURE_KIND,
        slot_id=exposure_id,
        values=_exposure_values(),
    )
    progress = capture_session_progress(CONTRACT, after_exposure, root=ROOT)
    assert progress["conditions"]["filled"] == 1
    assert progress["exposures"]["filled"] == 1
    assert progress["ready_for_complete_receipt_validation"] is False


def test_complete_incremental_packet_passes_existing_validator(tmp_path: Path) -> None:
    packet = build_receipt_template(CONTRACT, root=ROOT)
    for row in packet["common_condition_records"]:
        packet = update_capture_session_row(
            CONTRACT,
            packet,
            root=ROOT,
            kind=CONDITION_KIND,
            slot_id=row["condition_slot_id"],
            values=_condition_values(),
        )
    for row in packet["exposure_receipts"]:
        packet = update_capture_session_row(
            CONTRACT,
            packet,
            root=ROOT,
            kind=EXPOSURE_KIND,
            slot_id=row["exposure_slot_id"],
            values=_exposure_values(),
        )
    assert capture_session_progress(CONTRACT, packet, root=ROOT)[
        "ready_for_complete_receipt_validation"
    ]
    path = tmp_path / "packet.json"
    path.write_text(json.dumps(packet, sort_keys=True), encoding="utf-8")
    assert evaluate_receipts(CONTRACT, path, root=ROOT)["automatic_pass"] is True


def test_complete_batch_is_atomic_and_passes_existing_validator(
    tmp_path: Path,
) -> None:
    packet = build_receipt_template(CONTRACT, root=ROOT)
    updates = [
        {
            "kind": CONDITION_KIND,
            "slot_id": row["condition_slot_id"],
            "values": _condition_values(),
        }
        for row in reversed(packet["common_condition_records"])
    ] + [
        {
            "kind": EXPOSURE_KIND,
            "slot_id": row["exposure_slot_id"],
            "values": _exposure_values(),
        }
        for row in reversed(packet["exposure_receipts"])
    ]
    updated = update_capture_session_batch(CONTRACT, packet, root=ROOT, updates=updates)
    assert capture_session_progress(CONTRACT, updated, root=ROOT)[
        "ready_for_complete_receipt_validation"
    ]
    assert (
        capture_session_progress(CONTRACT, packet, root=ROOT)["conditions"]["filled"]
        == 0
    )
    path = tmp_path / "batch-packet.json"
    path.write_text(json.dumps(updated, sort_keys=True), encoding="utf-8")
    assert evaluate_receipts(CONTRACT, path, root=ROOT)["automatic_pass"] is True


def test_batch_rejects_duplicate_or_late_invalid_without_mutating_input() -> None:
    packet = build_receipt_template(CONTRACT, root=ROOT)
    condition_id = packet["common_condition_records"][0]["condition_slot_id"]
    exposure_id = packet["exposure_receipts"][0]["exposure_slot_id"]
    valid = {
        "kind": CONDITION_KIND,
        "slot_id": condition_id,
        "values": _condition_values(),
    }
    with pytest.raises(ThreeStockCaptureReceiptError, match="duplicate"):
        update_capture_session_batch(
            CONTRACT, packet, root=ROOT, updates=[valid, valid]
        )
    invalid_late = {
        "kind": EXPOSURE_KIND,
        "slot_id": exposure_id,
        "values": {**_exposure_values(), "shutter_seconds": 0.0},
    }
    with pytest.raises(ThreeStockCaptureReceiptError, match="shutter"):
        update_capture_session_batch(
            CONTRACT, packet, root=ROOT, updates=[valid, invalid_late]
        )
    assert (
        capture_session_progress(CONTRACT, packet, root=ROOT)["conditions"]["filled"]
        == 0
    )


def test_rejects_partial_unknown_or_duplicate_updates() -> None:
    packet = build_receipt_template(CONTRACT, root=ROOT)
    condition_id = packet["common_condition_records"][0]["condition_slot_id"]
    with pytest.raises(ThreeStockCaptureReceiptError, match="exact row"):
        update_capture_session_row(
            CONTRACT,
            packet,
            root=ROOT,
            kind=CONDITION_KIND,
            slot_id=condition_id,
            values={"display_device_id": "only-one-field"},
        )
    with pytest.raises(ThreeStockCaptureReceiptError, match="unknown"):
        update_capture_session_row(
            CONTRACT,
            packet,
            root=ROOT,
            kind=EXPOSURE_KIND,
            slot_id="missing",
            values=_exposure_values(),
        )
    filled = update_capture_session_row(
        CONTRACT,
        packet,
        root=ROOT,
        kind=CONDITION_KIND,
        slot_id=condition_id,
        values=_condition_values(),
    )
    with pytest.raises(ThreeStockCaptureReceiptError, match="already filled"):
        update_capture_session_row(
            CONTRACT,
            filled,
            root=ROOT,
            kind=CONDITION_KIND,
            slot_id=condition_id,
            values=_condition_values(),
        )


def test_cli_writes_new_packet_then_reports_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = build_receipt_template(CONTRACT, root=ROOT)
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    values_path = tmp_path / "values.json"
    values_path.write_text(json.dumps(_condition_values()), encoding="utf-8")
    updated_path = tmp_path / "updated.json"
    condition_id = packet["common_condition_records"][0]["condition_slot_id"]
    monkeypatch.setattr(
        "sys.argv",
        [
            "update-session",
            "--packet",
            str(packet_path),
            "--condition-id",
            condition_id,
            "--values",
            str(values_path),
            "--output",
            str(updated_path),
        ],
    )
    assert session_cli() == 0
    assert packet_path.read_text(encoding="utf-8") == json.dumps(packet)
    status_path = tmp_path / "status.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "update-session",
            "--packet",
            str(updated_path),
            "--status",
            "--output",
            str(status_path),
        ],
    )
    assert session_cli() == 0
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["conditions"]["filled"] == 1
    assert status["exposures"]["filled"] == 0


def test_cli_applies_batch_to_new_packet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = build_receipt_template(CONTRACT, root=ROOT)
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(
        json.dumps(
            [
                {
                    "kind": CONDITION_KIND,
                    "slot_id": packet["common_condition_records"][0][
                        "condition_slot_id"
                    ],
                    "values": _condition_values(),
                },
                {
                    "kind": EXPOSURE_KIND,
                    "slot_id": packet["exposure_receipts"][0]["exposure_slot_id"],
                    "values": _exposure_values(),
                },
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "batch-updated.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "update-session",
            "--packet",
            str(packet_path),
            "--batch",
            str(batch_path),
            "--output",
            str(output_path),
        ],
    )
    assert session_cli() == 0
    progress = capture_session_progress(
        CONTRACT, json.loads(output_path.read_bytes()), root=ROOT
    )
    assert progress["conditions"]["filled"] == 1
    assert progress["exposures"]["filled"] == 1
