#!/usr/bin/env python
"""Run the frozen U6.P2Y measured-interimage source feasibility audit."""

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

from src.eval.measured_interimage_source_audit import (
    audit_companion_pdf,
    sha256_file,
)

COMPANION_RELATIVE = Path(
    "outputs/source_recon/u6_p2y_measured_interimage_source_audit_v1/"
    "fairchild_berns_lester_shin_cic_1994.pdf"
)
REPORT_SCHEMA = "neuro-film.u6.p2y.measured-interimage-source-audit.v1"


def _canonical_sha(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def build_report(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    amendment = config["execution_amendments"][0]["facts"]
    companion = audit_companion_pdf(
        ROOT / COMPANION_RELATIVE,
        expected_bytes=int(amendment["companion_pdf_bytes"]),
        expected_sha256=str(amendment["companion_pdf_sha256"]),
    )
    requirements = config["audit_requirements"]
    local_thesis_present = False
    complete_rows = int(companion["complete_machine_readable_rgb_to_cmy_rows"])
    gates = {
        "official_landing_metadata_exact": True,
        "primary_thesis_browser_pages_exact": amendment[
            "primary_thesis_browser_page_count"
        ]
        == config["sources"][0]["expected_viewer_pages"],
        "local_thesis_pdf_available_for_numeric_claim": local_thesis_present,
        "companion_pdf_hash_and_parse_exact": True,
        "measurement_context_complete_in_companion": all(
            companion["anchor_presence"].values()
        ),
        "minimum_complete_numeric_rows": complete_rows
        >= int(requirements["minimum_complete_numeric_rows"]),
        "machine_readable_or_explicitly_tabulated_values": complete_rows > 0,
        "fit_or_render_count_zero": True,
    }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "node": "U6.P2Y",
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(config_path),
        "software_commit": _git_commit(),
        "question": config["question"],
        "source_access": {
            "primary_thesis": {
                "landing_url": config["sources"][0]["landing_url"],
                "pdf_url": config["sources"][0]["pdf_url"],
                "landing_and_browser_pdf_loaded": amendment[
                    "primary_thesis_landing_and_browser_pdf_loaded"
                ],
                "browser_page_count": amendment["primary_thesis_browser_page_count"],
                "local_pdf_available": local_thesis_present,
                "command_line_response": amendment[
                    "primary_thesis_command_line_response"
                ],
                "numeric_data_claim_allowed": False,
            },
            "official_companion": companion,
        },
        "numeric_data_feasibility": {
            "reported_dataset_colors": companion["reported_measurement_design"][
                "reported_rgb_to_cmy_dataset_colors"
            ],
            "published_complete_machine_readable_rows": complete_rows,
            "figure_ocr_or_manual_digitization_performed": False,
            "operator_fit_count": 0,
            "image_render_count": 0,
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "decision": "FAIL_CLOSED_NO_REUSABLE_ROW_LEVEL_DATA",
        "allowed_next": [
            "audit a materially distinct exact primary source under a new preregistration",
            "resume generic physical-inspired work only if the new mechanism is independently identified",
        ],
        "forbidden": config["forbidden_fallbacks"],
        "rights": config["rights"],
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = _canonical_sha(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p2y_measured_interimage_source_audit_v1.json",
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

