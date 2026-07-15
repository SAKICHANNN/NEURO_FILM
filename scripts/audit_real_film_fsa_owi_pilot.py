"""Download and audit the frozen 64-image FSA/OWI visual pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.fsa_owi_pilot import (  # noqa: E402
    download_pilot,
    render_contact_sheets,
    select_creator_balanced,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "real_film_fsa_owi_acquisition.json",
    )
    parser.add_argument(
        "--decision",
        type=Path,
        default=ROOT / "configs" / "real_film_fsa_owi_metadata_decision.json",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=ROOT / "outputs" / "real_film" / "fsa_owi_v1" / "metadata",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "real_film" / "fsa_owi_v1" / "pilot",
    )
    parser.add_argument("--seed", default="RF0.3-fsa-owi-pilot-v1")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    canonical_path = args.metadata_dir / "canonical_manifest.jsonl"
    report_path = args.metadata_dir / "report.json"
    if sha256(args.config) != decision["source_contract_sha256"]:
        raise ValueError("FSA/OWI source contract hash mismatch")
    if sha256(canonical_path) != decision["canonical_manifest_sha256"]:
        raise ValueError("FSA/OWI canonical manifest hash mismatch")
    if sha256(report_path) != decision["metadata_report_sha256"]:
        raise ValueError("FSA/OWI metadata report hash mismatch")
    records = [
        json.loads(line)
        for line in canonical_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    limits = config["phase_b_pilot"]
    selected = select_creator_balanced(
        records, count=int(limits["maximum_images"]), seed=args.seed
    )
    image_dir = args.output_dir / "images"
    rows, near_pairs = download_pilot(
        selected,
        image_dir=image_dir,
        maximum_total_bytes=int(limits["maximum_total_bytes"]),
        timeout_seconds=int(config["phase_a_metadata"]["timeout_seconds"]),
        max_retries=int(config["phase_a_metadata"]["max_retries"]),
    )
    sheets = render_contact_sheets(rows, args.output_dir / "contact_sheets")
    manifest_path = args.output_dir / "manifest.jsonl"
    manifest = b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        for row in rows
    )
    manifest_path.write_bytes(manifest)
    hashes = Counter(row["sha256"] for row in rows)
    report = {
        "schema_version": 1,
        "experiment_id": "RF0.3-fsa-owi-64-image-pilot",
        "software_commit": commit(),
        "config_sha256": sha256(args.config),
        "decision_sha256": sha256(args.decision),
        "canonical_manifest_sha256": sha256(canonical_path),
        "selection_seed": args.seed,
        "images": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "creator_counts": dict(sorted(Counter(row["creator_group"] for row in rows).items())),
        "format_counts": dict(sorted(Counter(row["format"] for row in rows).items())),
        "mode_counts": dict(sorted(Counter(row["mode"] for row in rows).items())),
        "icc_profile_counts": dict(
            sorted(Counter("present" if row["icc_profile_bytes"] else "absent" for row in rows).items())
        ),
        "exact_duplicate_payloads": sum(count - 1 for count in hashes.values() if count > 1),
        "near_duplicate_pairs_dhash_le_4": near_pairs,
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "contact_sheet_sha256": {str(path): sha256(path) for path in sheets},
        "manual_visual_audit": "pending",
        "training_executed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    pilot_report_path = args.output_dir / "report.json"
    pilot_report_path.write_bytes(encoded)
    print(json.dumps({
        "report": str(pilot_report_path),
        "report_sha256": hashlib.sha256(encoded).hexdigest(),
        "manifest_sha256": report["manifest_sha256"],
        "images": report["images"],
        "total_bytes": report["total_bytes"],
        "exact_duplicates": report["exact_duplicate_payloads"],
        "near_pairs": len(near_pairs),
        "contact_sheets": [str(path) for path in sheets],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
