from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.real_film.three_stock_acquisition import (
    MANIFEST_SCHEMA,
    ThreeStockAcquisitionError,
    compile_protocol,
    evaluate_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sf3_a0_three_stock_controlled_acquisition_v1.json"


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _manifest() -> dict[str, object]:
    stocks = {
        "fujifilm_velvia_50": ("e6_reversal", "direct_slide_neutral_scan"),
        "kodak_portra_400": ("c41_negative", "color_negative_neutral_scan"),
        "kodak_ektar_100": ("c41_negative", "color_negative_neutral_scan"),
    }
    rows: list[dict[str, object]] = []
    for stock_id, (process_type, interpretation_id) in stocks.items():
        for role, scene_count, roll_count, scanner_count in (
            ("development", 8, 2, 2),
            ("confirmation", 4, 1, 1),
        ):
            for roll_index in range(roll_count):
                roll_id = f"{role}-{stock_id}-roll-{roll_index}"
                process_id = f"{role}-{stock_id}-process-{roll_index}"
                for scene_index in range(scene_count):
                    scene_id = f"{role}-scene-{scene_index}"
                    frame_id = f"{role}-{stock_id}-{roll_index}-{scene_index}"
                    for scanner_index in range(scanner_count):
                        scanner_id = f"{role}-scanner-{scanner_index}"
                        row_id = f"{frame_id}-scan-{scanner_index}"
                        rows.append(
                            {
                                "row_id": row_id,
                                "stock_id": stock_id,
                                "role": role,
                                "scene_id": scene_id,
                                "scene_content_group": f"content-{role}-{scene_index}",
                                "film_frame_id": frame_id,
                                "capture_session_id": f"capture-{role}-{stock_id}-{roll_index}",
                                "source_owner_id": "project-owner",
                                "camera_system_id": "camera-system-v1",
                                "digital_reference_sha256": _hash(
                                    f"digital-{role}-{scene_index}"
                                ),
                                "capture_condition_sha256": _hash(
                                    f"condition-{role}-{scene_index}"
                                ),
                                "roll_id": roll_id,
                                "process_session_id": process_id,
                                "process_type": process_type,
                                "lab_id": f"controlled-lab-{role}",
                                "process_recipe_sha256": _hash(process_id),
                                "scanner_session_id": scanner_id,
                                "scanner_device_id": f"device-{role}-{scanner_index}",
                                "scanner_profile_sha256": _hash(scanner_id),
                                "scan_file_sha256": _hash(f"file-{row_id}"),
                                "scan_sample_sha256": _hash(f"sample-{row_id}"),
                                "alignment_evidence_sha256": _hash(
                                    f"alignment-{row_id}"
                                ),
                                "interpretation_id": interpretation_id,
                                "rights_record_sha256": _hash("owned-rights-v1"),
                                "rights_scope": "project_owned_internal_research_and_commercial_derivatives",
                                "rights_allow_internal_training": True,
                                "rights_allow_commercial_derivatives": True,
                                "rights_allow_released_weights": True,
                            }
                        )
    return {"schema": MANIFEST_SCHEMA, "rows": rows}


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_protocol_compiles_without_claiming_data() -> None:
    report = compile_protocol(CONFIG)
    assert report["decision"] == "READY_FOR_CONTROLLED_ACQUISITION_NO_DATA"
    assert report["minimum_rows_if_exactly_at_threshold"] == 108
    assert report["automatic_pass"] is False
    assert report["operator_fit_authority"] is False


def test_complete_group_held_manifest_passes(tmp_path: Path) -> None:
    report = evaluate_manifest(CONFIG, _write(tmp_path, _manifest()))
    assert report["row_count"] == 108
    assert report["automatic_pass"] is True
    assert report["pixel_reads"] == report["operator_fits"] == 0
    assert report["operator_fit_authority"] is False
    assert all(report["gates"].values())
    assert (
        report["decision"]
        == "OPEN_SF3_A1_FILE_PIXEL_ALIGNMENT_AND_RIGHTS_INTEGRITY_AUDIT"
    )


def test_scene_leakage_fails_holdout(tmp_path: Path) -> None:
    manifest = _manifest()
    for row in manifest["rows"]:  # type: ignore[index]
        if row["role"] == "confirmation" and row["scene_id"] == "confirmation-scene-0":
            row["scene_id"] = "development-scene-0"
    report = evaluate_manifest(CONFIG, _write(tmp_path, manifest))
    assert report["automatic_pass"] is False
    assert report["gates"]["cross_role_holdout"] is False
    assert report["gates"]["same_scene_controls"] is True


def test_rights_gap_fails_without_structural_error(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["rows"][0]["rights_allow_released_weights"] = False  # type: ignore[index]
    report = evaluate_manifest(CONFIG, _write(tmp_path, manifest))
    assert report["automatic_pass"] is False
    assert report["gates"]["rights_complete"] is False


def test_reused_frame_id_across_scenes_fails_role_support(tmp_path: Path) -> None:
    manifest = _manifest()
    rows = manifest["rows"]  # type: ignore[assignment]
    source_id = rows[0]["film_frame_id"]
    victim_id = rows[2]["film_frame_id"]
    for row in rows:
        if row["film_frame_id"] == victim_id:
            row["film_frame_id"] = source_id
    report = evaluate_manifest(CONFIG, _write(tmp_path, manifest))
    assert report["automatic_pass"] is False
    assert report["gates"]["role_support"] is False


def test_scanner_profile_drift_fails_role_support(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["rows"][0]["scanner_profile_sha256"] = _hash("drift")  # type: ignore[index]
    report = evaluate_manifest(CONFIG, _write(tmp_path, manifest))
    assert report["automatic_pass"] is False
    assert report["gates"]["role_support"] is False


def test_confirmation_scanner_device_reuse_fails_holdout(tmp_path: Path) -> None:
    manifest = _manifest()
    for row in manifest["rows"]:  # type: ignore[index]
        if row["role"] == "confirmation":
            row["scanner_device_id"] = "device-development-0"
    report = evaluate_manifest(CONFIG, _write(tmp_path, manifest))
    assert report["automatic_pass"] is False
    assert report["gates"]["cross_role_holdout"] is False


def test_duplicate_scan_identity_is_invalid(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["rows"][1]["scan_file_sha256"] = manifest["rows"][0][  # type: ignore[index]
        "scan_file_sha256"
    ]
    with pytest.raises(ThreeStockAcquisitionError, match="duplicate identity"):
        evaluate_manifest(CONFIG, _write(tmp_path, manifest))
