from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_u6_p2y_measured_interimage_source_audit import build_report
from src.eval.measured_interimage_source_audit import (
    MeasuredInterimageSourceError,
    audit_companion_pdf,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p2y_measured_interimage_source_audit_v1.json"
SOURCE = (
    ROOT
    / "outputs/source_recon/u6_p2y_measured_interimage_source_audit_v1/"
    "fairchild_berns_lester_shin_cic_1994.pdf"
)


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_exact_companion_exposes_design_but_not_row_values() -> None:
    facts = _config()["execution_amendments"][0]["facts"]
    audit = audit_companion_pdf(
        SOURCE,
        expected_bytes=facts["companion_pdf_bytes"],
        expected_sha256=facts["companion_pdf_sha256"],
    )
    assert audit["page_count"] == 5
    assert audit["reported_measurement_design"] == {
        "film": "Kodak Ektachrome 100 Plus Professional",
        "film_recorder": "Solitaire 8xp CRT film recorder",
        "spectral_absorptivity_exposures": 60,
        "single_channel_ramp_steps": 11,
        "modeling_exposures_described": 21,
        "reported_rgb_to_cmy_dataset_colors": 36,
        "operator_family": "three 1-D LUTs followed by a row-sum-one 3x3 matrix",
    }
    assert audit["candidate_lines_with_at_least_six_numeric_values"] == []
    assert audit["complete_machine_readable_rgb_to_cmy_rows"] == 0
    assert not audit["raw_row_level_measurements_published"]


def test_source_drift_fails_closed(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(SOURCE.read_bytes() + b"drift")
    facts = _config()["execution_amendments"][0]["facts"]
    with pytest.raises(MeasuredInterimageSourceError, match="byte count drift"):
        audit_companion_pdf(
            bad,
            expected_bytes=facts["companion_pdf_bytes"],
            expected_sha256=facts["companion_pdf_sha256"],
        )


def test_formal_report_closes_before_fit_or_digitization() -> None:
    report = build_report(CONFIG)
    assert report["decision"] == "FAIL_CLOSED_NO_REUSABLE_ROW_LEVEL_DATA"
    assert not report["all_gates_passed"]
    assert not report["gates"]["local_thesis_pdf_available_for_numeric_claim"]
    assert not report["gates"]["minimum_complete_numeric_rows"]
    assert report["numeric_data_feasibility"]["operator_fit_count"] == 0
    assert report["numeric_data_feasibility"]["image_render_count"] == 0
    assert not report["numeric_data_feasibility"][
        "figure_ocr_or_manual_digitization_performed"
    ]
