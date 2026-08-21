from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import src.real_film.three_stock_k1_file_runner as runner
from src.real_film.three_stock_k1_baseline import StockFrameSamples

ROOT = Path(__file__).resolve().parents[1]
K1_CONFIG = ROOT / "configs/sf3_a2_three_stock_k1_baseline_v1.json"
INTEGRITY_CONFIG = (
    ROOT / "configs/sf3_a1_three_stock_file_pixel_alignment_integrity_v1.json"
)
STOCKS = (
    "fujifilm_velvia_50",
    "kodak_portra_400",
    "kodak_ektar_100",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="ascii")


def test_load_aligned_rows_uses_verified_file_identities(tmp_path: Path) -> None:
    source = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)
    scan = np.flip(source, axis=1).copy()
    source_path = tmp_path / "data" / "source.png"
    scan_path = tmp_path / "data" / "scan.png"
    source_path.parent.mkdir(parents=True)
    assert cv2.imwrite(str(source_path), source[..., ::-1])
    assert cv2.imwrite(str(scan_path), scan[..., ::-1])
    evidence_path = tmp_path / "data" / "alignment.json"
    alignment_schema = "test-alignment.v1"
    _write_json(
        evidence_path,
        {
            "schema": alignment_schema,
            "row_id": "row-1",
            "digital_reference_sha256": "a" * 64,
            "scan_sample_sha256": "b" * 64,
            "homography_source_to_scan": np.eye(3).tolist(),
        },
    )
    ledger = {
        "rows": [
            {
                "alignment_evidence_path": "data/alignment.json",
                "digital_reference_path": "data/source.png",
                "scan_sample_path": "data/scan.png",
            }
        ]
    }
    manifest = {
        "rows": [
            {
                "row_id": "row-1",
                "stock_id": STOCKS[0],
                "role": "development",
                "scene_id": "scene-1",
                "film_frame_id": "frame-1",
                "roll_id": "roll-1",
                "digital_reference_sha256": "a" * 64,
                "scan_sample_sha256": "b" * 64,
            }
        ]
    }
    decode = json.loads(INTEGRITY_CONFIG.read_text(encoding="utf-8"))["decode"]
    rows = runner.load_aligned_rows(
        root=tmp_path,
        ledger=ledger,
        manifest=manifest,
        decode_contract=decode,
        alignment_schema=alignment_schema,
    )
    assert len(rows) == 1
    assert np.array_equal(rows[0].source_rgb, source)
    assert np.array_equal(rows[0].scan_rgb, scan)
    file_rows, paths = runner.load_aligned_file_rows(
        root=tmp_path,
        ledger=ledger,
        manifest=manifest,
        alignment_schema=alignment_schema,
    )
    assert len(file_rows) == 1
    assert file_rows[0].source_shape == source.shape
    assert file_rows[0].scan_shape == scan.shape
    assert paths["row-1"] == (source_path, scan_path)


def test_integrity_failure_stops_before_sampling_or_fit(
    tmp_path: Path, monkeypatch
) -> None:
    ledger = tmp_path / "ledger.json"
    manifest = tmp_path / "manifest.json"
    _write_json(ledger, {"rows": []})
    _write_json(manifest, {"rows": []})
    monkeypatch.setattr(
        runner,
        "evaluate_integrity",
        lambda *args, **kwargs: {
            "automatic_pass": False,
            "stable_evidence_id": "integrity-fail",
            "decision": "RETAIN_THREE_STOCK_DATA_GAP",
        },
    )
    monkeypatch.setattr(
        runner,
        "load_aligned_file_rows",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("sampling reached")),
    )
    report = runner.evaluate_files(
        root=ROOT,
        integrity_contract_path=INTEGRITY_CONFIG,
        k1_contract_path=K1_CONFIG,
        ledger_path=ledger,
        manifest_path=manifest,
    )
    assert report["automatic_pass"] is False
    assert report["paired_sampling_executed"] is False
    assert report["operator_fits"] == 0


def test_success_reports_actual_roll_dependent_fit_count(
    tmp_path: Path, monkeypatch
) -> None:
    ledger = tmp_path / "ledger.json"
    manifest = tmp_path / "manifest.json"
    _write_json(ledger, {"rows": []})
    _write_json(manifest, {"rows": []})
    monkeypatch.setattr(
        runner,
        "evaluate_integrity",
        lambda *args, **kwargs: {
            "automatic_pass": True,
            "stable_evidence_id": "integrity-pass",
        },
    )
    monkeypatch.setattr(
        runner, "load_aligned_file_rows", lambda **kwargs: ([], {})
    )
    sample = np.full((4, 3), 0.25, dtype=np.float64)
    development = {
        stock: [
            StockFrameSamples(
                scene_id=f"scene-{roll}",
                frame_id=f"frame-{roll}",
                roll_id=f"roll-{roll}",
                source=sample,
                target=sample,
            )
            for roll in range(roll_count)
        ]
        for stock, roll_count in zip(STOCKS, (2, 3, 4), strict=True)
    }
    confirmation = {stock: [] for stock in STOCKS}
    monkeypatch.setattr(
        runner,
        "extract_common_paired_samples_streaming",
        lambda rows, config, **kwargs: (
            development,
            confirmation,
            {"row_count": 0},
        ),
    )
    monkeypatch.setattr(
        runner,
        "evaluate_k1",
        lambda *args, **kwargs: {
            "automatic_pass": True,
            "decision": "PASS",
            "stable_evidence_id": "k1-pass",
        },
    )
    report = runner.evaluate_files(
        root=ROOT,
        integrity_contract_path=INTEGRITY_CONFIG,
        k1_contract_path=K1_CONFIG,
        ledger_path=ledger,
        manifest_path=manifest,
    )
    assert report["automatic_pass"] is True
    assert report["operator_fits"] == (2 + 3 + 4) * 3 + 3
