from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_u6_p2z_emulating_emulsion_author_package_audit import build_report
from src.eval.emulating_emulsion_author_package_audit import (
    EmulatingEmulsionPackageError,
    audit_author_page,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p2z_emulating_emulsion_author_package_audit_v1.json"
PAGE = (
    ROOT
    / "outputs/source_recon/u6_p2z_emulating_emulsion_author_package_audit_v1/author_page.html"
)


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_author_page_claims_code_but_exposes_no_package_link() -> None:
    facts = _config()["execution_amendments"][0]["facts"]
    audit = audit_author_page(
        PAGE,
        expected_bytes=facts["author_page_bytes"],
        expected_sha256=facts["author_page_sha256"],
    )
    assert audit["claims_provided_source_code"]
    assert audit["explicit_repository_links"] == []
    assert audit["downloadable_code_or_data_links"] == []
    assert audit["published_complete_paired_patch_rows"] == 0
    assert audit["published_exact_fitted_parameter_values"] == 0


def test_author_asset_drift_fails_closed(tmp_path: Path) -> None:
    bad = tmp_path / "page.html"
    bad.write_bytes(PAGE.read_bytes() + b"drift")
    facts = _config()["execution_amendments"][0]["facts"]
    with pytest.raises(EmulatingEmulsionPackageError, match="byte count drift"):
        audit_author_page(
            bad,
            expected_bytes=facts["author_page_bytes"],
            expected_sha256=facts["author_page_sha256"],
        )


def test_formal_report_closes_before_fit_or_render() -> None:
    report = build_report(CONFIG)
    assert report["decision"] == (
        "FAIL_CLOSED_PAPER_ONLY_NO_REPRODUCIBLE_AUTHOR_PACKAGE"
    )
    assert not report["all_gates_passed"]
    assert not report["gates"]["explicit_author_link_to_source_code"]
    assert not report["gates"][
        "exact_fitted_parameter_bundle_or_complete_paired_patch_rows"
    ]
    assert report["reproducibility_facts"]["operator_fit_count"] == 0
    assert report["reproducibility_facts"]["image_render_count"] == 0
    assert report["reproducibility_facts"]["visual_review_count"] == 0
    assert report["source_access"]["full_paper"]["page_count"] == 8
    assert report["source_access"]["github"]["algorithm_repository_found"] is False
