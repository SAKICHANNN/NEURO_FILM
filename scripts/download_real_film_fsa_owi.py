"""Enumerate and gate the LOC-identified FSA/OWI Commons mirror."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.fsa_owi import (  # noqa: E402
    build_canonical_subset,
    evaluate_metadata_gate,
    fetch_category_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "real_film_fsa_owi_acquisition.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "real_film" / "fsa_owi_v1" / "metadata",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    records, api_evidence = fetch_category_records(config)
    raw_gate = evaluate_metadata_gate(records, config)
    canonical, canonical_gate = build_canonical_subset(records, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    manifest = b"".join(
        (json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        for record in records
    )
    manifest_path.write_bytes(manifest)
    canonical_path = args.output_dir / "canonical_manifest.jsonl"
    canonical_manifest = b"".join(
        (json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        for record in canonical
    )
    canonical_path.write_bytes(canonical_manifest)
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "metadata_snapshot_date": config["metadata_snapshot_date"],
        "software_commit": commit(),
        "config_sha256": sha256(args.config),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "canonical_manifest_sha256": hashlib.sha256(canonical_manifest).hexdigest(),
        "api_calls": len(api_evidence),
        "api_response_evidence": api_evidence,
        "raw_gate": raw_gate,
        "canonical_gate": canonical_gate,
        "phase_b_images_downloaded": 0,
        "claim_ceiling": config["claim_ceiling"],
    }
    report_path = args.output_dir / "report.json"
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    report_path.write_bytes(encoded)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "report_sha256": hashlib.sha256(encoded).hexdigest(),
                "manifest": str(manifest_path),
                "manifest_sha256": report["manifest_sha256"],
                "decision": canonical_gate["decision"],
                "raw_decision": raw_gate["decision"],
                "records": raw_gate["records"],
                "canonical_records": canonical_gate["canonical_records"],
                "unique_creators": raw_gate["unique_creators"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if canonical_gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
