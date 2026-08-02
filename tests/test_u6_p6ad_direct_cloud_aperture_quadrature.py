from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.direct_cloud_aperture_quadrature import (
    DirectCloudQuadratureAuditError,
    evaluate_direct_cloud_aperture_quadrature,
    load_contract,
)
from src.film_physics.developed_structure import build_colour_dye_cloud_context
from src.film_physics.direct_cloud_aperture import (
    DirectCloudApertureError,
    concentric_golden_angle_disk_points,
    render_direct_cloud_aperture,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6ad_direct_cloud_aperture_quadrature_v1.json"
PARENT_REPORT = (
    ROOT / "outputs/experiments/u6_p6ac_presampling_dye_cloud_scan_v1/report_run1.json"
)
DECISION = ROOT / "configs/u6_p6ad_direct_cloud_aperture_quadrature_decision_v1.json"


def test_disk_points_are_bounded_and_repeat_exact() -> None:
    first = concentric_golden_angle_disk_points(64)
    second = concentric_golden_angle_disk_points(64)
    assert np.array_equal(first, second)
    assert np.max(np.sum(first * first, axis=1)) < 1.0
    with pytest.raises(DirectCloudApertureError):
        concentric_golden_angle_disk_points(0)


def test_small_direct_render_is_partition_exact() -> None:
    target = np.full((3, 4, 3), 0.2, dtype=np.float64)
    context = build_colour_dye_cloud_context(
        target,
        radius_um_cmy=(1.8, 2.2, 2.6),
        mark_optical_density_cmy=(0.16, 0.20, 0.24),
        output_zoom=1,
        output_pixel_pitch_um=6.35,
        monte_carlo_samples=2,
        seed=7,
    )
    full = render_direct_cloud_aperture(
        context,
        aperture_diameter_um=12.5,
        sample_count=16,
        point_chunk_size=128,
    )
    partitioned = render_direct_cloud_aperture(
        context,
        aperture_diameter_um=12.5,
        sample_count=16,
        point_chunk_size=128,
        row_partition=1,
    )
    assert np.array_equal(full, partitioned)
    assert np.all((full > 0.0) & (full <= 1.0))


def test_contract_rejects_sample_count_mutation(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["quadrature"]["candidate_samples"] = 512
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DirectCloudQuadratureAuditError, match="contract drift"):
        load_contract(path)


@pytest.mark.skipif(
    not PARENT_REPORT.is_file(), reason="P6AC reference report unavailable"
)
def test_frozen_evaluator_repeats_without_intermediate_raster() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_direct_cloud_aperture_quadrature(contract, ROOT)
    second = evaluate_direct_cloud_aperture_quadrature(contract, ROOT)
    assert first == second
    assert contract["quadrature"]["intermediate_raster_allowed"] is False
    assert first["metrics"]["repeat_error"] == 0.0
    assert first["metrics"]["partition_error"] == 0.0


@pytest.mark.skipif(
    not PARENT_REPORT.is_file(), reason="P6AC reference report unavailable"
)
def test_frozen_decision_matches_replay() -> None:
    report = evaluate_direct_cloud_aperture_quadrature(load_contract(CONTRACT), ROOT)
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["stable_evidence_id"] == report["stable_evidence_id"]
    assert decision["automatic_pass"] is False
    assert decision["sample_count_rescue_allowed"] is False
