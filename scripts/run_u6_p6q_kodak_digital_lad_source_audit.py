#!/usr/bin/env python
"""Run the frozen U6.P6Q Kodak Digital LAD source audit."""

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

from src.eval.kodak_digital_lad_source_audit import (
    audit_html,
    audit_pdf,
    audit_zip_inventory,
    sha256_file,
)

SOURCE_ROOT = ROOT / "data/physical_scanner/kodak_digital_lad_v1"
REPORT_SCHEMA = "neuro-film.u6.p6q.kodak-digital-lad-source-audit.v1"


def _canonical_sha(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def build_report(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sources = {row["role"]: row for row in config["primary_sources"]}
    facts = config["execution_amendments"][0]["facts"]
    required_urls = [row["url"] for row in config["primary_sources"][1:]]
    page = audit_html(
        SOURCE_ROOT / "official_page.html",
        expected_bytes=int(facts["official_page_bytes"]),
        expected_sha256=str(facts["official_page_sha256"]),
        required_urls=required_urls,
    )
    guide_row = sources["Digital LAD user guide H-387"]
    guide = audit_pdf(
        SOURCE_ROOT / "H-387.pdf",
        expected_bytes=int(guide_row["expected_bytes"]),
        expected_sha256=str(facts["guide_sha256"]),
        expected_pages=int(guide_row["expected_pages"]),
        required_phrases={
            "title": "KODAK Digital LAD Test Image",
            "negative_equation": "Printing Density = 0.002 * CV",
            "lad_code": "445 445 445",
            "negative_mode": "Negative Mode 5/2254",
            "interpositive_mode": "IP Mode 5/2254",
            "status_m": "Status M density above D-min",
        },
    )
    spec_row = sources["Cineon file-format description"]
    cineon_spec = audit_pdf(
        SOURCE_ROOT / "Cineon-File-Format-Description.pdf",
        expected_bytes=int(spec_row["expected_bytes"]),
        expected_sha256=str(facts["cineon_spec_sha256"]),
        expected_pages=int(spec_row["expected_pages"]),
        required_phrases={
            "title": "Image File Format Proposal for Digital Pictures",
            "pixel_semantics": "What the pixel values represent",
            "linear_metric": "relation of code value to data metric is linear",
        },
    )
    maximum_member = int(config["source_gate"]["no_archive_member_above_bytes"])
    maximum_ratio = float(config["source_gate"]["zip_expansion_ratio_bounded_by"])
    dpx_row = sources["Digital LAD DPX archive"]
    dpx = audit_zip_inventory(
        SOURCE_ROOT / "Digital-LAD-DPX.zip",
        expected_bytes=int(dpx_row["expected_bytes"]),
        expected_sha256=str(facts["dpx_zip_sha256"]),
        maximum_member_bytes=maximum_member,
        maximum_expansion_ratio=maximum_ratio,
    )
    cin_row = sources["Digital LAD Cineon archive"]
    cineon = audit_zip_inventory(
        SOURCE_ROOT / "Digital-LAD-Cineon.zip",
        expected_bytes=int(cin_row["expected_bytes"]),
        expected_sha256=str(facts["cineon_zip_sha256"]),
        maximum_member_bytes=maximum_member,
        maximum_expansion_ratio=maximum_ratio,
    )
    inventory_safe = all(
        row["unique_member_names"]
        and row["all_member_paths_safe"]
        and row["expansion_ratio_within_limit"]
        and row["largest_member_within_limit"]
        for row in (dpx, cineon)
    )
    gates = {
        "official_asset_hashes_and_sizes_exact": True,
        "official_page_links_all_four_assets": True,
        "pdf_page_counts_and_required_semantics_exact": True,
        "zip_member_paths_relative_unique_and_non_symlink": all(
            row["unique_member_names"] and row["all_member_paths_safe"]
            for row in (dpx, cineon)
        ),
        "zip_expansion_ratios_within_limit": all(
            row["expansion_ratio_within_limit"] for row in (dpx, cineon)
        ),
        "all_archive_members_within_frozen_byte_limit": all(
            row["largest_member_within_limit"] for row in (dpx, cineon)
        ),
        "image_headers_and_geometry_recorded": False,
        "machine_readable_lad_patch_found": False,
        "dpx_cineon_pixel_equivalence": False,
        "operator_fit_render_and_visual_review_count_zero": True,
    }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "node": "U6.P6Q",
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(config_path),
        "software_commit": _git_commit(),
        "question": config["question"],
        "source_access": {
            "official_page": page,
            "guide": guide,
            "cineon_spec": cineon_spec,
            "dpx_archive": dpx,
            "cineon_archive": cineon,
        },
        "execution": {
            "archive_inventory_safe": inventory_safe,
            "archive_member_decompress_count": 0,
            "image_header_parse_count": 0,
            "pixel_decode_count": 0,
            "operator_fit_count": 0,
            "image_render_count": 0,
            "visual_review_count": 0,
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "decision": "FAIL_CLOSED_ARCHIVE_MEMBER_EXCEEDS_FROZEN_LIMIT",
        "retained_evidence": "exact official H-387 neutral recorder/print semantics and bounded ZIP inventory only; no archive member was decompressed",
        "allowed_next": "open a separate guide-table-only exact code-to-density primitive without using either ZIP image",
        "forbidden": config["forbidden"],
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = _canonical_sha(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p6q_kodak_digital_lad_source_audit_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.config.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    sys.stdout.buffer.write(encoded.encode("utf-8"))


if __name__ == "__main__":
    main()
