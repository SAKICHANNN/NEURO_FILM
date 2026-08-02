from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.vision3_16mm_tiff_integrity import (
    Vision3TiffIntegrityError,
    canonical_json,
    evaluate_tiff_integrity,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bu7_vision3_16mm_tiff_integrity_v1.json"


def test_contract_rejects_contact_threshold_drift(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["decode"]["minimum_matched_gradient_ncc"] = 0.9
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Vision3TiffIntegrityError, match="contract drift"):
        load_contract(changed)


def test_tiff_audit_is_exact() -> None:
    config = load_contract(CONFIG)
    first = evaluate_tiff_integrity(ROOT, config)
    second = evaluate_tiff_integrity(ROOT, config)
    assert canonical_json(first) == canonical_json(second)


def test_exact_tiff_route_closes_on_frozen_rgb16_gate() -> None:
    report = evaluate_tiff_integrity(ROOT, load_contract(CONFIG))
    assert report["gate_pass"] is False
    assert report["downloaded_file_count"] == 6
    assert report["downloaded_total_bytes"] == 76_583_832
    assert report["gate_results"][
        "all_six_contact_order_matches_pass_ncc_and_wrong_margin"
    ]
    assert not report["gate_results"][
        "all_six_tiffs_decode_rgb16_and_meet_minimum_geometry"
    ]
    assert {row["mode"] for row in report["file_rows"]} == {"RGBA"}
    assert {tuple(row["bits_per_sample"]) for row in report["file_rows"]} == {
        (8, 8, 8, 8)
    }
    assert {tuple(row["alpha_values"]) for row in report["file_rows"]} == {(255,)}
    assert {row["icc_profile_sha256"] for row in report["file_rows"]} == {None}
    assert report["decision"] == "close_exact_public_tiff_integrity_route"
