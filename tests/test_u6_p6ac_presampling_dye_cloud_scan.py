from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.presampling_dye_cloud_scan import (
    PresamplingDyeCloudAuditError,
    evaluate_presampling_dye_cloud_scan,
    load_contract,
)
from src.film_physics.presampling_dye_cloud_scan import (
    render_presampling_dye_cloud_scan,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6ac_presampling_dye_cloud_scan_v1.json"
PARENT = ROOT / "configs/u6_p6ab_circular_scanner_aperture_4000dpi_decision_v1.json"
DECISION = ROOT / "configs/u6_p6ac_presampling_dye_cloud_scan_decision_v1.json"


def test_contract_rejects_zoom_mutation(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["material"]["candidate_zoom"] = 16
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PresamplingDyeCloudAuditError, match="contract drift"):
        load_contract(path)


def test_core_rejects_invalid_density() -> None:
    with pytest.raises(ValueError, match="invalid presampling"):
        render_presampling_dye_cloud_scan(
            np.full((3, 4, 3), -1.0),
            target_pixel_pitch_um=6.35,
            candidate_zoom=8,
            reference_zoom=16,
            aperture_diameter_um=12.5,
            aperture_subpixels_per_axis=32,
            radius_um_cmy=(1.8, 2.2, 2.6),
            mark_optical_density_cmy=(0.16, 0.20, 0.24),
            monte_carlo_samples=2,
            seed=1,
        )


@pytest.mark.skipif(not PARENT.is_file(), reason="P6AB decision unavailable")
def test_frozen_evaluator_repeats_and_stays_reference_only() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_presampling_dye_cloud_scan(contract, ROOT)
    second = evaluate_presampling_dye_cloud_scan(contract, ROOT)
    assert first == second
    assert 0.0 < first["metrics"]["minimum_transmittance"]
    assert first["metrics"]["maximum_transmittance"] <= 1.0
    assert "no measured film" in first["claim_ceiling"].lower()


@pytest.mark.skipif(not PARENT.is_file(), reason="P6AB decision unavailable")
def test_frozen_decision_matches_replay() -> None:
    report = evaluate_presampling_dye_cloud_scan(load_contract(CONTRACT), ROOT)
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["stable_evidence_id"] == report["stable_evidence_id"]
    assert decision["automatic_pass"] is False
    assert decision["failed_gates"] == [
        "maximum_candidate_rmse",
        "maximum_candidate_p95_absolute_error",
    ]
