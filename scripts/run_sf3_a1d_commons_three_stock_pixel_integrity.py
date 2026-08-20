"""Download and audit the frozen author-balanced Commons three-stock pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.commons_stock_pilot import (
    atomic_json,
    audit_download_manifest,
    download_selected_rows,
    render_contact_sheets,
    sha256_file,
)
from src.real_film.commons_three_stock_pixel import (
    build_selection,
    load_contract,
    load_metadata_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/sf3_a1d_commons_three_stock_pixel_integrity_v1.json",
    )
    args = parser.parse_args()
    contract = load_contract(args.contract)
    metadata_path = ROOT / contract["metadata_report"]
    metadata = load_metadata_report(metadata_path, contract)
    selection = build_selection(metadata, contract)
    selection_path = ROOT / contract["selection_manifest"]
    selection_sha = atomic_json(selection_path, selection)

    checkpoint_path = ROOT / contract["download_checkpoint"]
    prior = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.is_file()
        else None
    )
    limits = contract["download_limits"]
    download_root = ROOT / contract["download_root"]
    manifest = download_selected_rows(
        selection["rows"],
        root=download_root,
        config=contract,
        prior_manifest=prior,
        timeout_seconds=int(limits["timeout_seconds"]),
        retries=int(limits["retries"]),
        retry_backoff_seconds=float(limits["retry_backoff_seconds"]),
        request_interval_seconds=float(limits["request_interval_seconds"]),
        checkpoint_path=checkpoint_path,
    )
    manifest_path = ROOT / contract["download_manifest"]
    manifest_sha = atomic_json(manifest_path, manifest)
    report = audit_download_manifest(manifest, root=download_root, config=contract)
    report.update(
        {
            "schema": "neuro-film.sf3-a1d-commons-three-stock-pixel-integrity-report.v1",
            "experiment_id": contract["experiment_id"],
            "config_sha256": sha256_file(args.contract),
            "metadata_report_sha256": contract["metadata_report_sha256"],
            "selection_manifest_sha256": selection_sha,
            "download_manifest_sha256": manifest_sha,
            "operator_fits": 0,
            "operator_fitting_allowed": False,
        }
    )
    report["contact_sheets"] = render_contact_sheets(
        report["file_records"],
        root=download_root,
        output_dir=ROOT / contract["contact_sheet_root"],
    )
    source_gates = all(
        row["learning_source_gate_passed"] for row in report["stock_source_gates"].values()
    )
    report["automatic_integrity_pass"] = bool(report["integrity_passed"] and source_gates)
    report["decision"] = (
        contract["decision_if_integrity_pass"]
        if report["automatic_integrity_pass"]
        else contract["decision_if_integrity_fail"]
    )
    report_sha = atomic_json(ROOT / contract["audit_report"], report)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "files": report["files"],
                "bytes": report["bytes"],
                "integrity_passed": report["integrity_passed"],
                "stock_source_gates": report["stock_source_gates"],
                "report_sha256": report_sha,
                "contact_sheets": len(report["contact_sheets"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["automatic_integrity_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
