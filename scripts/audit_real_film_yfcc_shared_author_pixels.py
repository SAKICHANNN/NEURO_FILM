"""Audit the frozen SF1.3A pixels and render stock-separated contact sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.commons_stock_pilot import render_contact_sheets  # noqa: E402
from src.real_film.yfcc_shared_author_pixels import audit_shared_author_pixels  # noqa: E402
from src.real_film.yfcc_stock_source import atomic_json  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "real_film_yfcc_shared_author_pixel_v1.json",
    )
    parser.add_argument(
        "--visual-verdict",
        choices=("pending", "pass_no_confirmed_severe", "fail_confirmed_severe"),
        default="pending",
    )
    parser.add_argument("--visual-notes", default="")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["download_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = audit_shared_author_pixels(
        manifest, root=ROOT / config["download_root"], pixel_config=config
    )
    report["evidence_identity"] = {
        "config_sha256": _sha(args.config),
        "download_manifest_sha256": _sha(manifest_path),
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "audit_utc": datetime.now(timezone.utc).isoformat(),
        "acquisition_timestamp_limit": (
            "the acquisition manifest records each live-page result and local file lineage but "
            "the first implementation omitted an exact per-request UTC field"
        ),
    }
    report["contact_sheets"] = render_contact_sheets(
        report["file_records"],
        root=ROOT / config["download_root"],
        output_dir=ROOT / config["contact_sheet_root"],
    )
    risk_rows = sorted(
        report["file_records"],
        key=lambda row: max(
            float(row["black_fraction_luma_le_1"]),
            float(row["white_fraction_luma_ge_254"]),
        ),
        reverse=True,
    )[:12]
    report["full_resolution_visual_risk_cases"] = [
        {
            "local_path": row["local_path"],
            "film_stock_id": row["film_stock_id"],
            "author_uid": row["author_uid"],
            "black_fraction_luma_le_1": row["black_fraction_luma_le_1"],
            "white_fraction_luma_ge_254": row["white_fraction_luma_ge_254"],
            "reason": "largest automatic endpoint-fraction triage; not an artifact verdict",
        }
        for row in risk_rows
    ]
    report["autonomous_visual_adjudication"] = {
        "verdict": args.visual_verdict,
        "reviewed_contact_sheets": len(report["contact_sheets"]),
        "reviewed_full_resolution_risk_cases": len(risk_rows),
        "confirmed_severe_count": 0 if args.visual_verdict == "pass_no_confirmed_severe" else None,
        "notes": args.visual_notes,
        "claim_scope": "autonomous Codex visual evidence, not population preference or stock identifiability",
    }
    report["visual_adjudication_status"] = args.visual_verdict
    digest = atomic_json(ROOT / config["audit_report"], report)
    print(
        json.dumps(
            {
                "files": report["files"],
                "integrity_passed": report["integrity_passed"],
                "exact_duplicate_groups": len(report["exact_duplicate_groups"]),
                "near_duplicate_pairs": len(report["near_duplicate_pairs_dhash_le_4"]),
                "bilateral_authors": report["bilateral_pixel_author_count"],
                "contact_sheets": len(report["contact_sheets"]),
                "report_sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
