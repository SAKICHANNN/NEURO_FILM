"""Audit frozen BlueNeg stock-pilot pixels for decode integrity and support."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.stock_pilot_integrity import (  # noqa: E402
    StockPilotIntegrityError,
    audit_stock_pilot_integrity,
    render_stock_pilot_contact_sheets,
    write_stock_pilot_integrity_report,
)
from src.roll2film.blueneg_download import sha256_file  # noqa: E402


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decision",
        type=Path,
        default=ROOT / "configs" / "real_film_stock_pilot_download_decision.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "real_film_stock_pilot_integrity_audit.json",
    )
    args = parser.parse_args()
    decision = _load_json(args.decision)
    config = _load_json(args.config)

    acquisition_path = ROOT / config["acquisition_manifest"]
    download_report_path = ROOT / config["download_report"]
    metadata_report_path = ROOT / config["metadata_report"]
    frames_path = ROOT / config["frames_jsonl"]
    rolls_path = ROOT / config["rolls_jsonl"]
    download_root = (ROOT / decision["download_root"]).resolve()
    output_report = ROOT / config["output_report"]
    contact_dir = output_report.parent / "contact_sheets"

    for path, expected in (
        (acquisition_path, decision["acquisition_manifest_sha256"]),
        (download_report_path, decision["download_report_sha256"]),
        (metadata_report_path, decision["metadata_report_sha256"]),
    ):
        actual = sha256_file(path)
        if actual != expected:
            raise StockPilotIntegrityError(
                f"hash mismatch for {path}: {actual} != {expected}"
            )

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    report = audit_stock_pilot_integrity(
        download_root=download_root,
        acquisition=_load_json(acquisition_path),
        download_report=_load_json(download_report_path),
        metadata_report=_load_json(metadata_report_path),
        frame_rows=_load_jsonl(frames_path),
        roll_rows=_load_jsonl(rolls_path),
        software_commit=commit,
        dhash_threshold=int(config.get("dhash_threshold", 4)),
    )
    sheets = render_stock_pilot_contact_sheets(
        download_root=download_root,
        file_records=report["file_records"],
        output_dir=contact_dir,
    )
    report["contact_sheets"] = sheets
    report_hash = write_stock_pilot_integrity_report(output_report, report)
    summary = {
        key: report[key]
        for key in (
            "schema_version",
            "audit_id",
            "software_commit",
            "acquisition_file_count",
            "acquisition_bytes",
            "passed",
            "checks",
            "eligibility_by_stock",
            "claim_ceiling",
            "border_mask_status",
            "visual_adjudication_status",
        )
    }
    summary["contact_sheet_count"] = len(sheets)
    summary["report_path"] = str(output_report.as_posix())
    summary["report_sha256"] = report_hash
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StockPilotIntegrityError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
