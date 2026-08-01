from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.controlled_reversal_source_audit import (
    DECISION,
    ControlledReversalSourceAuditError,
    evaluate_controlled_reversal_source_audit,
    load_contract,
    write_report,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p6y_controlled_reversal_source_audit_v1.json"


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return evaluate_controlled_reversal_source_audit(ROOT, load_contract(CONTRACT))


def test_retained_public_deposit_identity_is_exact(
    report: dict[str, object],
) -> None:
    assert report["record"]["title_exact"] is True
    assert report["record"]["handle_exact"] is True
    assert report["record"]["declared_inventory_exact"] is True
    assert report["record"]["open_artifact_count"] == 2
    assert report["record"]["only_pdf_and_zip"] is True
    assert report["checks"]["retained_payload_integrity"] is True


def test_support_zip_contains_only_administrative_pdfs(
    report: dict[str, object],
) -> None:
    archive = report["support_zip"]
    assert archive["member_count"] == 5
    assert archive["pdf_member_count"] == 4
    assert archive["substantive_pdf_member_count"] == 2
    assert archive["member_names_unique"] is True
    assert archive["all_member_paths_safe"] is True
    assert archive["pixel_payload_members"] == []
    assert archive["code_members"] == []


def test_thesis_methods_are_preserved_without_inventing_process_metadata(
    report: dict[str, object],
) -> None:
    anchors = report["thesis"]["method_anchors"]
    assert anchors["controlled_dataset"] is True
    assert anchors["two_mock_paintings"] is True
    assert anchors["two_stocks"] is True
    assert anchors["two_illuminants"] is True
    assert anchors["multiple_exposures"] is True
    assert anchors["120_reversal_film"] is True
    assert anchors["multispectral_film_scan"] is True
    assert anchors["painting_hyperspectral_ground_truth"] is True
    assert anchors["e6_process"] is False
    assert report["thesis"]["separate_data_repository_linked"] is False


def test_missing_machine_readable_pixels_closes_before_pixel_work(
    report: dict[str, object],
) -> None:
    assert report["automatic_pass"] is False
    assert report["decision"] == DECISION
    assert report["pixel_payload_count"] == 0
    assert report["pixel_decode_count"] == 0
    assert report["operator_fit_count"] == 0
    assert report["render_count"] == 0
    assert report["visual_review_count"] == 0
    assert report["checks"]["machine_readable_controlled_pixel_payloads"] is False
    assert report["checks"]["pixel_payload_rights_explicit"] is False


def test_report_encoding_is_repeatable(
    report: dict[str, object], tmp_path: Path
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    assert write_report(report, first) == write_report(report, second)
    assert first.read_bytes() == second.read_bytes()


def test_contract_execution_fact_drift_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["execution_amendment"]["thesis_pdf_sha256"] = "0" * 64
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ControlledReversalSourceAuditError, match="contract drift"):
        load_contract(path)
