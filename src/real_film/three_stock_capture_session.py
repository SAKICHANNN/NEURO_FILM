"""Copy-on-write field recording for the frozen SF3.A0N receipt packet."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.real_film.three_stock_capture_receipts import (
    ThreeStockCaptureReceiptError,
    build_receipt_template,
    build_single_stock_receipt_template,
)

CONDITION_KIND = "condition"
EXPOSURE_KIND = "exposure"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _template(contract_path: Path, *, root: Path, stock: str | None) -> dict[str, Any]:
    if stock is None:
        return build_receipt_template(contract_path, root=root)
    return build_single_stock_receipt_template(contract_path, root=root, stock=stock)


def _finite(value: Any, *, positive: bool = False) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and (not positive or float(value) > 0.0)
    )


def _validate_condition(row: Mapping[str, Any]) -> None:
    for key in ("display_device_id", "camera_system_id", "lens_id", "framing_id"):
        if not isinstance(row[key], str) or not row[key]:
            raise ThreeStockCaptureReceiptError(f"invalid common-condition {key}")
    if not _SHA256.fullmatch(str(row["display_profile_sha256"])):
        raise ThreeStockCaptureReceiptError("invalid display profile identity")
    for key in ("display_luminance_cd_m2", "aperture_f_number", "focus_distance_m"):
        if not _finite(row[key], positive=True):
            raise ThreeStockCaptureReceiptError(f"invalid common-condition {key}")
    if (
        not _finite(row["ambient_illuminance_lux"])
        or row["ambient_illuminance_lux"] < 0
    ):
        raise ThreeStockCaptureReceiptError("invalid ambient illuminance")


def _validate_exposure(row: Mapping[str, Any]) -> None:
    if not _finite(row["shutter_seconds"], positive=True):
        raise ThreeStockCaptureReceiptError("invalid shutter duration")
    for key in ("meter_reading_ev100", "exposure_compensation_ev"):
        if not _finite(row[key]):
            raise ThreeStockCaptureReceiptError(f"invalid exposure {key}")
    if (
        not isinstance(row["meter_id"], str)
        or not row["meter_id"]
        or not _SHA256.fullmatch(str(row["meter_calibration_sha256"]))
    ):
        raise ThreeStockCaptureReceiptError("invalid exposure meter identity")


def _rows(packet: Mapping[str, Any], kind: str) -> list[dict[str, Any]]:
    key = "common_condition_records" if kind == CONDITION_KIND else "exposure_receipts"
    value = packet.get(key)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ThreeStockCaptureReceiptError("capture session inventory drift")
    return value


def _id_field(kind: str) -> str:
    if kind == CONDITION_KIND:
        return "condition_slot_id"
    if kind == EXPOSURE_KIND:
        return "exposure_slot_id"
    raise ThreeStockCaptureReceiptError("unsupported capture session row kind")


def _validate_partial(packet: Mapping[str, Any], template: Mapping[str, Any]) -> None:
    if set(packet) != set(template):
        raise ThreeStockCaptureReceiptError("capture session packet field drift")
    for key in set(template) - {"common_condition_records", "exposure_receipts"}:
        if packet[key] != template[key]:
            raise ThreeStockCaptureReceiptError("capture session parent drift")
    for kind in (CONDITION_KIND, EXPOSURE_KIND):
        id_field = _id_field(kind)
        expected_rows = _rows(template, kind)
        observed_rows = _rows(packet, kind)
        if len(expected_rows) != len(observed_rows):
            raise ThreeStockCaptureReceiptError("capture session row-count drift")
        expected_by_id = {row[id_field]: row for row in expected_rows}
        observed_ids: set[str] = set()
        for row in observed_rows:
            slot = row.get(id_field)
            expected = expected_by_id.get(slot)
            if expected is None or slot in observed_ids or set(row) != set(expected):
                raise ThreeStockCaptureReceiptError(
                    "capture session slot or field drift"
                )
            observed_ids.add(slot)
            for key, value in expected.items():
                if value is not None and row[key] != value:
                    raise ThreeStockCaptureReceiptError(
                        "capture session immutable field drift"
                    )
            mutable = [key for key, value in expected.items() if value is None]
            filled = [row[key] is not None for key in mutable]
            if any(filled) and not all(filled):
                raise ThreeStockCaptureReceiptError(
                    "capture session row is partially filled"
                )
            if all(filled):
                (_validate_condition if kind == CONDITION_KIND else _validate_exposure)(
                    row
                )
        if observed_ids != set(expected_by_id):
            raise ThreeStockCaptureReceiptError("capture session slot coverage drift")


def capture_session_progress(
    contract_path: Path,
    packet: Mapping[str, Any],
    *,
    root: Path,
    stock: str | None = None,
) -> dict[str, Any]:
    """Return exact fill progress without accepting a partially filled row."""

    template = _template(contract_path, root=root, stock=stock)
    _validate_partial(packet, template)
    result: dict[str, Any] = {
        "schema": "neuro-film.sf3-a0n-capture-session-progress.v1",
        "stock": stock,
    }
    ready = True
    for kind, plural in ((CONDITION_KIND, "conditions"), (EXPOSURE_KIND, "exposures")):
        id_field = _id_field(kind)
        expected_by_id = {row[id_field]: row for row in _rows(template, kind)}
        unfilled = []
        for row in _rows(packet, kind):
            mutable = [
                key
                for key, value in expected_by_id[row[id_field]].items()
                if value is None
            ]
            if not all(row[key] is not None for key in mutable):
                unfilled.append(row[id_field])
        result[plural] = {
            "total": len(expected_by_id),
            "filled": len(expected_by_id) - len(unfilled),
            "remaining": len(unfilled),
            "next_unfilled_id": unfilled[0] if unfilled else None,
        }
        ready &= not unfilled
    result["ready_for_complete_receipt_validation"] = ready
    return result


def update_capture_session_row(
    contract_path: Path,
    packet: Mapping[str, Any],
    *,
    root: Path,
    kind: str,
    slot_id: str,
    values: Mapping[str, Any],
    stock: str | None = None,
) -> dict[str, Any]:
    """Return a new packet with one complete mutable row recorded."""

    template = _template(contract_path, root=root, stock=stock)
    _validate_partial(packet, template)
    updated = copy.deepcopy(packet)
    id_field = _id_field(kind)
    expected_by_id = {row[id_field]: row for row in _rows(template, kind)}
    expected = expected_by_id.get(slot_id)
    if expected is None:
        raise ThreeStockCaptureReceiptError("unknown capture session slot")
    mutable = {key for key, value in expected.items() if value is None}
    if set(values) != mutable or any(value is None for value in values.values()):
        raise ThreeStockCaptureReceiptError(
            "capture session update must fill one exact row"
        )
    target = next(row for row in _rows(updated, kind) if row[id_field] == slot_id)
    if any(target[key] is not None for key in mutable):
        raise ThreeStockCaptureReceiptError("capture session row is already filled")
    target.update(values)
    (_validate_condition if kind == CONDITION_KIND else _validate_exposure)(target)
    _validate_partial(updated, template)
    return updated


__all__ = [
    "CONDITION_KIND",
    "EXPOSURE_KIND",
    "capture_session_progress",
    "update_capture_session_row",
]
