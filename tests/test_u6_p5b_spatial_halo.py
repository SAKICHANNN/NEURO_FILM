from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.physical_spatial_halo import (
    _bounded_edge_metrics,
    evaluate_spatial_halo,
    load_contract,
)
from src.eval.physical_spatial_response import _slanted_edge


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p5b_spatial_halo_audit_v1.json"
P5A = ROOT / "configs" / "u6_p5a_spatial_response_primitives_v1.json"
SENSITOMETRY = ROOT / "configs" / "u2_2a_sensitometry_primitive_v1.json"


def test_bounded_edge_metrics_never_reports_above_nyquist() -> None:
    values, distance = _slanted_edge((64, 128), 5.0, 0.1, 0.8)
    metrics = _bounded_edge_metrics(
        values[..., 0],
        distance,
        pixel_pitch_um=1.0,
        nyquist_cycles_per_mm=500.0,
    )
    for key in ("mtf50_cycles_per_mm", "mtf10_cycles_per_mm"):
        value = metrics[key]
        assert value is None or value <= 500.0


def test_p5b_contract_rejects_wrong_schema() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["schema"] = "wrong"
    temporary = ROOT / "tmp" / "test_u6_p5b_wrong.json"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    try:
        try:
            load_contract(temporary)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong schema was accepted")
    finally:
        temporary.unlink(missing_ok=True)


def test_p5b_frozen_profile_exposes_undershoot_risk(tmp_path: Path) -> None:
    report = evaluate_spatial_halo(
        load_contract(CONTRACT),
        json.loads(P5A.read_text(encoding="utf-8")),
        json.loads(SENSITOMETRY.read_text(encoding="utf-8")),
        diagnostic_path=tmp_path / "diagnostic.png",
    )
    assert report["automatic_pass"] is False
    assert report["stable_evidence_id"] == (
        "45d4c82e055011ff6f4f6478f07d4dffd3aead6b0bcda118a4a52e447d8218ad"
    )
    assert report["decisions"]["undershoot"] is False
    assert report["decisions"]["absolute_halo"] is False
    assert report["decisions"]["zero_adjacency"] is True
    assert report["decisions"]["nyquist_reporting"] is True
    assert (tmp_path / "diagnostic.png").is_file()
