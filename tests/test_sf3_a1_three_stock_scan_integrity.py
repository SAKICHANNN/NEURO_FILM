from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile

from src.preprocess import srgb_icc_profile, srgb_icc_profile_sha256
from src.real_film.three_stock_acquisition import (
    LEDGER_SCHEMA,
    compile_acquisition_ledger,
)
from src.real_film.three_stock_scan_integrity import (
    ThreeStockScanIntegrityError,
    build_alignment_evidence,
    decode_integer_rgb,
    decode_scan_integer_rgb,
    evaluate,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sf3_a1_three_stock_file_pixel_alignment_integrity_v1.json"
A0_CONFIG = ROOT / "configs/sf3_a0_three_stock_controlled_acquisition_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="ascii")


def _image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(16, 240, size=(72, 96, 3), dtype=np.uint8)
    cv2.circle(image, (24, 24), 13, (250, 12, 140), 2)
    cv2.line(image, (5, 60), (90, 9), (10, 245, 80), 2)
    return image


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), rgb[..., ::-1])


def _fixture(
    tmp_path: Path, *, bad_rights: bool = False, bad_alignment: bool = False
) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    shutil.copyfile(A0_CONFIG, repo / "configs" / A0_CONFIG.name)
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    contract["alignment"].update(
        {
            "maximum_features": 500,
            "minimum_good_matches": 4,
            "minimum_ransac_inliers": 4,
            "maximum_median_inlier_reprojection_error_px": 0.1,
        }
    )
    contract["decode"].update({"required_scan_width": 96, "required_scan_height": 72})
    contract_path = repo / "configs" / CONFIG.name
    _write_json(contract_path, contract)

    data = repo / "data" / "sf3_a1_fixture"
    data.mkdir(parents=True)
    rights = {
        "schema": contract["record_schemas"]["rights"],
        "rights_record_id": "owned-rights-v1",
        "source_owner_id": "project-owner",
        "rights_scope": "project_owned_internal_research_and_commercial_derivatives",
        "allow_internal_training": True,
        "allow_commercial_derivatives": True,
        "allow_released_weights": not bad_rights,
    }
    rights_path = data / "rights.json"
    _write_json(rights_path, rights)

    rows: list[dict[str, object]] = []
    stocks = {
        "fujifilm_velvia_50": ("e6_reversal", "direct_slide_neutral_scan"),
        "kodak_portra_400": ("c41_negative", "color_negative_neutral_scan"),
        "kodak_ektar_100": ("c41_negative", "color_negative_neutral_scan"),
    }
    source_paths: dict[tuple[str, int], Path] = {}
    for role, count in (("development", 8), ("confirmation", 4)):
        for scene_index in range(count):
            source = data / f"source-{role}-{scene_index}.png"
            _write_rgb(
                source, _image(scene_index + (0 if role == "development" else 100))
            )
            source_paths[(role, scene_index)] = source

    row_index = 0
    for stock_id, (process_type, interpretation_id) in stocks.items():
        for role, scene_count, roll_count, scanner_count in (
            ("development", 8, 2, 2),
            ("confirmation", 4, 1, 1),
        ):
            for roll_index in range(roll_count):
                roll_id = f"{role}-{stock_id}-roll-{roll_index}"
                process_id = f"{role}-{stock_id}-process-{roll_index}"
                process_path = data / f"{process_id}.json"
                _write_json(process_path, {"process_id": process_id})
                for scene_index in range(scene_count):
                    source_path = source_paths[(role, scene_index)]
                    condition_path = data / f"condition-{role}-{scene_index}.json"
                    _write_json(condition_path, {"role": role, "scene": scene_index})
                    frame_id = f"{role}-{stock_id}-{roll_index}-{scene_index}"
                    for scanner_index in range(scanner_count):
                        scanner_id = f"{role}-scanner-{scanner_index}"
                        scanner_path = data / f"{scanner_id}.json"
                        _write_json(scanner_path, {"scanner_id": scanner_id})
                        row_id = f"{frame_id}-scan-{scanner_index}"
                        scan_path = data / f"scan-{row_index:03d}.tif"
                        scan = cv2.convertScaleAbs(
                            cv2.imread(str(source_path), cv2.IMREAD_COLOR),
                            alpha=1.0,
                            beta=(row_index % 7) - 3,
                        )
                        scan[0, 0] = (
                            row_index,
                            (row_index * 17) % 256,
                            (row_index * 31) % 256,
                        )
                        profile = srgb_icc_profile()
                        tifffile.imwrite(
                            scan_path,
                            (scan[..., ::-1].astype(np.uint16) * 257),
                            photometric="rgb",
                            planarconfig="contig",
                            metadata=None,
                            extratags=[(34675, "B", len(profile), profile, False)],
                        )
                        alignment_path = data / f"alignment-{row_index:03d}.json"
                        row = {
                            "row_id": row_id,
                            "stock_id": stock_id,
                            "role": role,
                            "scene_id": f"{role}-scene-{scene_index}",
                            "scene_content_group": f"content-{role}-{scene_index}",
                            "film_frame_id": frame_id,
                            "capture_session_id": f"capture-{role}-{stock_id}-{roll_index}",
                            "source_owner_id": "project-owner",
                            "camera_system_id": "camera-system-v1",
                            "roll_id": roll_id,
                            "process_session_id": process_id,
                            "process_type": process_type,
                            "lab_id": f"controlled-lab-{role}",
                            "scanner_session_id": scanner_id,
                            "scanner_device_id": f"device-{role}-{scanner_index}",
                            "interpretation_id": interpretation_id,
                            "rights_scope": "project_owned_internal_research_and_commercial_derivatives",
                            "rights_allow_internal_training": True,
                            "rights_allow_commercial_derivatives": True,
                            "rights_allow_released_weights": True,
                            "digital_reference_path": source_path.relative_to(
                                repo
                            ).as_posix(),
                            "capture_condition_record_path": condition_path.relative_to(
                                repo
                            ).as_posix(),
                            "process_recipe_record_path": process_path.relative_to(
                                repo
                            ).as_posix(),
                            "scanner_profile_record_path": scanner_path.relative_to(
                                repo
                            ).as_posix(),
                            "scan_file_path": scan_path.relative_to(repo).as_posix(),
                            "scan_sample_path": scan_path.relative_to(repo).as_posix(),
                            "alignment_evidence_path": alignment_path.relative_to(
                                repo
                            ).as_posix(),
                            "rights_record_path": rights_path.relative_to(
                                repo
                            ).as_posix(),
                        }
                        evidence_row = {
                            **row,
                            "digital_reference_sha256": _sha(source_path),
                            "scan_sample_sha256": _sha(scan_path),
                        }
                        source_rgb = cv2.imread(str(source_path), cv2.IMREAD_COLOR)[
                            ..., ::-1
                        ]
                        scan_rgb = cv2.imread(str(scan_path), cv2.IMREAD_COLOR)[
                            ..., ::-1
                        ]
                        evidence = build_alignment_evidence(
                            evidence_row,
                            source_rgb,
                            scan_rgb,
                            alignment=contract["alignment"],
                            schema=contract["record_schemas"]["alignment"],
                        )
                        if bad_alignment and row_index == 0:
                            evidence["homography_source_to_scan"][0][2] += 1.0
                        _write_json(alignment_path, evidence)
                        rows.append(row)
                        row_index += 1
    assert len(rows) == 108
    ledger_path = data / "ledger.json"
    _write_json(ledger_path, {"schema": LEDGER_SCHEMA, "rows": rows})
    manifest = compile_acquisition_ledger(
        repo / "configs" / A0_CONFIG.name, ledger_path, root=repo
    )
    manifest_path = data / "manifest.json"
    _write_json(manifest_path, manifest)
    return repo, contract_path, ledger_path, manifest_path


def test_contract_binds_current_a0() -> None:
    _, contract = load_contract(CONFIG, root=ROOT)
    assert contract["status"] == "FROZEN_BEFORE_PHYSICAL_SCAN_READ"
    assert (
        contract["decode"]["required_rgb16_tiff_icc_profile_sha256"]
        == srgb_icc_profile_sha256()
    )
    assert (
        contract["decode"]["required_scan_width"],
        contract["decode"]["required_scan_height"],
    ) == (3000, 2000)


def test_rgb16_tiff_requires_bound_canonical_srgb_icc(tmp_path: Path) -> None:
    _, contract = load_contract(CONFIG, root=ROOT)
    pixels = np.arange(8 * 10 * 3, dtype=np.uint16).reshape(8, 10, 3)
    profile = srgb_icc_profile()
    valid = tmp_path / "valid.tif"
    tifffile.imwrite(
        valid,
        pixels,
        photometric="rgb",
        planarconfig="contig",
        metadata=None,
        extratags=[(34675, "B", len(profile), profile, False)],
    )
    np.testing.assert_array_equal(decode_integer_rgb(valid, contract["decode"]), pixels)

    unprofiled = tmp_path / "unprofiled.tif"
    tifffile.imwrite(
        unprofiled,
        pixels,
        photometric="rgb",
        planarconfig="contig",
        metadata=None,
    )
    np.testing.assert_array_equal(
        decode_integer_rgb(unprofiled, contract["decode"]), pixels
    )

    small_contract = {
        **contract["decode"],
        "required_scan_width": 10,
        "required_scan_height": 8,
    }
    np.testing.assert_array_equal(
        decode_scan_integer_rgb(valid, small_contract), pixels
    )
    with pytest.raises(ThreeStockScanIntegrityError, match="canonical sRGB ICC"):
        decode_scan_integer_rgb(unprofiled, small_contract)
    png_scan = tmp_path / "scan.png"
    assert cv2.imwrite(str(png_scan), pixels.astype(np.uint8)[..., ::-1])
    with pytest.raises(ThreeStockScanIntegrityError, match="RGB16 TIFF"):
        decode_scan_integer_rgb(png_scan, small_contract)
    uint8_scan = tmp_path / "scan.tif"
    tifffile.imwrite(
        uint8_scan,
        pixels.astype(np.uint8),
        photometric="rgb",
        planarconfig="contig",
        metadata=None,
    )
    with pytest.raises(ThreeStockScanIntegrityError, match="RGB16 TIFF"):
        decode_scan_integer_rgb(uint8_scan, small_contract)
    with pytest.raises(ThreeStockScanIntegrityError, match="geometry"):
        decode_scan_integer_rgb(valid, {**small_contract, "required_scan_height": 7})

    rotated = tmp_path / "rotated.tif"
    tifffile.imwrite(
        rotated,
        pixels,
        photometric="rgb",
        planarconfig="contig",
        metadata=None,
        extratags=[
            (274, "H", 1, 6, False),
            (34675, "B", len(profile), profile, False),
        ],
    )
    with pytest.raises(ThreeStockScanIntegrityError, match="orientation"):
        decode_scan_integer_rgb(rotated, small_contract)


def test_complete_fixture_passes_without_operator_fit(tmp_path: Path) -> None:
    repo, contract, ledger, manifest = _fixture(tmp_path)
    report = evaluate(contract, ledger, manifest, root=repo)
    assert report["row_count"] == 108
    assert report["pixel_reads"] == 216
    assert report["operator_fits"] == 0
    assert report["automatic_pass"] is True
    assert all(report["gates"].values())


@pytest.mark.parametrize("failure", ["rights", "alignment"])
def test_row_evidence_mismatch_fails_closed(tmp_path: Path, failure: str) -> None:
    repo, contract, ledger, manifest = _fixture(
        tmp_path,
        bad_rights=failure == "rights",
        bad_alignment=failure == "alignment",
    )
    report = evaluate(contract, ledger, manifest, root=repo)
    assert report["automatic_pass"] is False
    assert report["operator_fit_authority"] is False
    assert report["decision"].startswith("RETAIN_THREE_STOCK_DATA_GAP")


def test_manifest_must_be_exact_compilation(tmp_path: Path) -> None:
    repo, contract, ledger, manifest = _fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="ascii"))
    payload["rows"][0]["scan_sample_sha256"] = "0" * 64
    _write_json(manifest, payload)
    with pytest.raises(ThreeStockScanIntegrityError, match="does not match ledger"):
        evaluate(contract, ledger, manifest, root=repo)
