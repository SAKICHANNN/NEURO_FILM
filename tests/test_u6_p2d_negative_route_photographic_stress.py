from __future__ import annotations

import json
from pathlib import Path

from src.eval.negative_route_photographic_stress import (
    REFERENCE_GAUGE_SCHEMA,
    SCHEMA,
    evaluate,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2d_negative_route_photographic_stress_v1.json"
REFERENCE_GAUGE_CONTRACT = (
    ROOT / "configs/u6_p2e_reference_gauge_negative_stress_v1.json"
)


def test_contract_excludes_closed_slide_and_unproved_bw() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["schema"] == SCHEMA
    assert contract["routes"] == [
        "color_negative_neutral_scan",
        "color_negative_print",
    ]
    assert "slide_direct_scan" not in contract["routes"]
    assert "bw_developer_scan" not in contract["routes"]


def test_fixed_visual_smoke_is_repeat_exact_and_bounded(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    manifest = json.loads(
        (ROOT / contract["input"]["manifest"]).read_text(encoding="utf-8")
    )
    selected = [manifest[0]]
    smoke = json.loads(json.dumps(contract))
    smoke["input"]["expected_rows"] = len(selected)
    smoke["input"]["expected_camera_makes"] = len(
        {row["make"] for row in selected}
    )
    smoke["visual_protocol"]["fixed_ids"] = [selected[0]["id"]]
    first = evaluate(
        smoke,
        selected,
        root=ROOT,
        contact_sheet_path=tmp_path / "a.png",
    )
    second = evaluate(
        smoke,
        selected,
        root=ROOT,
        contact_sheet_path=tmp_path / "b.png",
    )
    assert first == second
    assert first["contact_sheet_sha256"] == second["contact_sheet_sha256"]
    assert first["decisions"]["finite_bounded"] is True
    assert first["decisions"]["repeat"] is True


def test_reference_gauge_contract_is_frozen_before_render() -> None:
    contract = json.loads(REFERENCE_GAUGE_CONTRACT.read_text(encoding="utf-8"))
    assert contract["schema"] == REFERENCE_GAUGE_SCHEMA
    assert contract["pipeline"]["pseudo_exposure_scale"] == 1.0
    assert (
        contract["pipeline"]["print_maximum_relative_layer_exposure"] == 1.0
    )
    assert contract["automatic_gates"]["near_white_threshold"] == 0.98
    assert (
        contract["automatic_gates"][
            "maximum_per_image_near_white_fraction"
        ]
        == 0.10
    )
    assert (
        contract["automatic_gates"][
            "minimum_per_image_luma_p95_minus_p05"
        ]
        == 0.10
    )


def test_reference_gauge_smoke_reports_tone_collapse_metrics(
    tmp_path: Path,
) -> None:
    contract = json.loads(REFERENCE_GAUGE_CONTRACT.read_text(encoding="utf-8"))
    manifest = json.loads(
        (ROOT / contract["input"]["manifest"]).read_text(encoding="utf-8")
    )
    selected = [manifest[0]]
    smoke = json.loads(json.dumps(contract))
    smoke["input"]["expected_rows"] = 1
    smoke["input"]["expected_camera_makes"] = 1
    smoke["visual_protocol"]["fixed_ids"] = [selected[0]["id"]]
    report = evaluate(
        smoke,
        selected,
        root=ROOT,
        contact_sheet_path=tmp_path / "reference-gauge.png",
    )
    assert report["schema"].endswith(
        "u6_p2e_reference_gauge_negative_stress_report.v1"
    )
    assert "per_image_near_white" in report["decisions"]
    assert "population_near_white" in report["decisions"]
    assert "photographic_luma_range" in report["decisions"]
    for route in ("negative", "print"):
        assert "maximum_near_white_fraction" in report["population"][route]
        assert "minimum_luma_p95_minus_p05" in report["population"][route]
