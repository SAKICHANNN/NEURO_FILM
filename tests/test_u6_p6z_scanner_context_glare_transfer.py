from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.scanner_context_glare_transfer import (
    FAIL_DECISION,
    ScannerContextGlareTransferError,
    evaluate_scanner_context_glare_transfer,
    load_contract,
    load_source_table,
    write_report,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p6z_scanner_context_glare_transfer_v1.json"
SOURCE = ROOT / "configs" / "u6_p6z_scanner_context_glare_table_v1.json"


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return evaluate_scanner_context_glare_transfer(
        load_contract(CONTRACT), load_source_table(ROOT, SOURCE)
    )


def test_source_table_and_pdf_are_exact(report: dict[str, object]) -> None:
    assert report["source_pdf"]["pdf_bytes"] == 2_590_747
    assert report["source_pdf"]["pdf_pages"] == 9
    assert len(report["derived_table_rows"]) == 24
    assert sum(row["eligible"] for row in report["derived_table_rows"]) == 20
    exclusions = {
        (row["polarity"], row["scanner_id"]): row["exclusion_reason"]
        for row in report["derived_table_rows"]
        if not row["eligible"]
    }
    assert exclusions == {
        ("positive-scan-negative-image", 3): "undefined-table-value",
        ("positive-scan-negative-image", 6): "negative-difference",
        ("positive-scan-negative-image", 7): "undefined-table-value",
        ("negative-scan-positive-image", 7): "undefined-table-value",
    }


def test_same_hardware_operator_is_development_only_and_bounded(
    report: dict[str, object],
) -> None:
    assert len(report["confirmation_rows"]) == 9
    assert report["metrics"]["minimum_development_fraction"] >= 0.0
    assert report["metrics"]["maximum_development_fraction"] <= 0.25
    assert report["checks"]["development_fraction_bounds"] is True
    assert report["checks"]["finite_outputs"] is True
    assert all(
        row["scanner_id"] in {2, 3, 9, 10, 12} for row in report["confirmation_rows"]
    )


def test_transfer_signal_is_real_but_tail_gate_closes(
    report: dict[str, object],
) -> None:
    metrics = report["metrics"]
    assert metrics["hardware_transfer_win_rate_vs_no_context"] == pytest.approx(
        7.0 / 9.0
    )
    assert metrics["hardware_transfer_win_rate_vs_global"] == pytest.approx(7.0 / 9.0)
    assert metrics["median_error_improvement_vs_no_context"] > 0.88
    assert metrics["median_error_improvement_vs_global"] > 0.73
    assert metrics["median_error_improvement_vs_wrong_group"] > 0.83
    assert metrics["confirmation_p95_absolute_bc_error"] == pytest.approx(
        0.035878426442663196
    )
    assert report["failed_gates"] == ["confirmation_p95_absolute_bc_error"]
    assert report["automatic_pass"] is False
    assert report["decision"] == FAIL_DECISION


def test_claim_boundary_excludes_pixels_and_product(report: dict[str, object]) -> None:
    assert report["pixel_decode_count"] == 0
    assert report["image_render_count"] == 0
    assert report["visual_review_count"] == 0
    assert "not a scanner calibration" in report["claim_ceiling"]


def test_report_is_byte_repeatable(report: dict[str, object], tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    assert write_report(report, first) == write_report(report, second)
    assert first.read_bytes() == second.read_bytes()


def test_contract_and_source_drift_fail_closed(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["automatic_gates"]["maximum_confirmation_p95_absolute_bc_error"] = 0.04
    drifted_contract = tmp_path / "contract.json"
    drifted_contract.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ScannerContextGlareTransferError, match="contract drift"):
        load_contract(drifted_contract)

    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    source["rows"][0]["cg"][0] = 1.0
    drifted_source = tmp_path / "source.json"
    drifted_source.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ScannerContextGlareTransferError, match="source table drift"):
        load_source_table(ROOT, drifted_source)
