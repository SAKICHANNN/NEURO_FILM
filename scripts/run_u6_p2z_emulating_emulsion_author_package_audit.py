#!/usr/bin/env python
"""Run the frozen U6.P2Z Emulating Emulsion author-package audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.emulating_emulsion_author_package_audit import (
    audit_author_page,
    audit_pdf,
    sha256_file,
)

SOURCE_ROOT = ROOT / "outputs/source_recon/u6_p2z_emulating_emulsion_author_package_audit_v1"
FULL_PAPER = ROOT / "outputs/source_recon/emulating_emulsion_siggraph2025/full_paper.pdf"
REPORT_SCHEMA = "neuro-film.u6.p2z.emulating-emulsion-author-package-audit.v1"


def _canonical_sha(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def build_report(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    facts = config["execution_amendments"][0]["facts"]
    page = audit_author_page(
        SOURCE_ROOT / "author_page.html",
        expected_bytes=int(facts["author_page_bytes"]),
        expected_sha256=str(facts["author_page_sha256"]),
    )
    abstract = audit_pdf(
        SOURCE_ROOT / "siggraph_abstract.pdf",
        expected_bytes=int(facts["abstract_bytes"]),
        expected_sha256=str(facts["abstract_sha256"]),
        expected_pages=int(facts["abstract_pages"]),
        required_phrases={
            "title": "Emulating Emulsion: A Compact Physically-Based Model for Film Colour",
            "single_roll": "single film roll",
            "raw_patch_rows": "3168 RAW patch pairs",
            "parameter_count": "30 (36 including bias terms) parameters",
        },
    )
    poster = audit_pdf(
        SOURCE_ROOT / "siggraph_poster_srgb.pdf",
        expected_bytes=int(facts["poster_bytes"]),
        expected_sha256=str(facts["poster_sha256"]),
        expected_pages=int(facts["poster_pages"]),
        required_phrases={
            "title": "A Compact Physically-Based Model for Film Colour",
            "parameter_count": "30 Parameters",
            "single_roll": "single roll of film",
        },
    )
    full_paper = audit_pdf(
        FULL_PAPER,
        expected_bytes=int(facts["preexisting_full_paper_bytes"]),
        expected_sha256=str(facts["preexisting_full_paper_sha256"]),
        expected_pages=8,
        required_phrases={
            "title": "Emulating Emulsion: A Compact Physically-Based Model for Film Colour",
            "total_patch_pairs": "4620patch pairs",
            "unique_patch_rows": "3168unique patches",
            "single_roll": "single 36-exposure roll",
            "cross_validation": "5-fold cross-validated average RMSE",
        },
    )

    explicit_repository_links = list(page["explicit_repository_links"])
    for source in (abstract, poster, full_paper):
        explicit_repository_links.extend(source["explicit_code_repository_links"])
    complete_rows = int(page["published_complete_paired_patch_rows"])
    exact_parameters = int(page["published_exact_fitted_parameter_values"])
    requirements = config["audit_requirements"]
    gates = {
        "author_page_paper_and_poster_hashes_exact": True,
        "preexisting_full_paper_hash_and_parse_exact": True,
        "explicit_author_link_to_source_code": bool(explicit_repository_links),
        "exact_algorithm_repository_revision": False,
        "software_license": False,
        "runnable_operator_implementation": False,
        "exact_fitted_parameter_bundle_or_complete_paired_patch_rows": (
            exact_parameters >= 30
            or complete_rows
            >= int(requirements["minimum_complete_paired_patch_rows_if_parameters_absent"])
        ),
        "data_or_parameter_rights_for_local_reproduction": False,
        "capture_process_scan_and_color_domain_lineage_complete": False,
        "published_fit_evaluation_role_manifest": False,
        "operator_fit_render_and_visual_review_count_zero": True,
    }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "node": "U6.P2Z",
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(config_path),
        "software_commit": _git_commit(),
        "question": config["question"],
        "source_access": {
            "author_page": page,
            "abstract": abstract,
            "poster": poster,
            "full_paper": full_paper,
            "github": {
                "profile": facts["explicit_github_profile"],
                "public_repository_count": facts["github_public_repository_count"],
                "website_repository": facts["website_repository"],
                "website_repository_commit": facts["website_repository_commit"],
                "website_repository_license": facts["website_repository_license"],
                "emulating_emulsion_page_blob": facts["emulating_emulsion_page_blob"],
                "algorithm_repository_found": False,
            },
            "new_full_paper_download": {
                "content_length": facts["full_paper_content_length"],
                "retained": facts["new_full_paper_download_retained"],
                "reason": "exceeds frozen new-acquisition byte cap; exact pre-existing copy audited instead",
            },
        },
        "reproducibility_facts": {
            "author_page_claims_provided_source_code": page[
                "claims_provided_source_code"
            ],
            "explicit_algorithm_source_repository_links": explicit_repository_links,
            "complete_paired_patch_rows_published": complete_rows,
            "exact_fitted_parameter_values_published": exact_parameters,
            "operator_fit_count": 0,
            "image_render_count": 0,
            "visual_review_count": 0,
            "single_roll_only": True,
            "grain_halation_mtf_model_included": False,
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "decision": "FAIL_CLOSED_PAPER_ONLY_NO_REPRODUCIBLE_AUTHOR_PACKAGE",
        "retained_evidence": "current best paired compact equation-family and capture-design baseline; public author assets do not expose the implementation, fitted values, paired rows, rights or complete process/split lineage required for reproduction",
        "allowed_next": [
            "retain AO9 as the already-closed clean-room capacity control on display proxies",
            "continue a materially distinct rights-cleared identifying-data or physical-mechanism leaf",
        ],
        "forbidden": config["forbidden_fallbacks"],
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = _canonical_sha(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p2z_emulating_emulsion_author_package_audit_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.config.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
