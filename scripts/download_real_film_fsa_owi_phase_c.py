"""Acquire the bounded 558-record grouped FSA/OWI Phase-C corpus."""

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

from src.real_film.fsa_owi_dataset import (  # noqa: E402
    merge_pixel_evidence,
    prepare_download_records,
    seed_verified_pilot,
)
from src.real_film.fsa_owi_pilot import download_pilot  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "real_film_fsa_owi_phase_c.json"
    )
    parser.add_argument(
        "--grouped-manifest",
        type=Path,
        default=ROOT / "outputs" / "real_film" / "fsa_owi_v1" / "grouping" / "manifest.jsonl",
    )
    parser.add_argument(
        "--pilot-manifest",
        type=Path,
        default=ROOT / "outputs" / "real_film" / "fsa_owi_v1" / "pilot" / "manifest.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "real_film" / "fsa_owi_v1" / "phase_c",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    grouping_decision = ROOT / "configs" / "real_film_fsa_owi_grouping_decision.json"
    if sha256(grouping_decision) != config["grouping_decision_sha256"]:
        raise ValueError("grouping decision hash mismatch")
    if sha256(args.grouped_manifest) != config["grouped_manifest_sha256"]:
        raise ValueError("grouped manifest hash mismatch")
    grouped = [json.loads(line) for line in args.grouped_manifest.read_text(encoding="utf-8").splitlines() if line]
    if len(grouped) != int(config["expected_records"]):
        raise ValueError("unexpected grouped record count")
    prepared = prepare_download_records(grouped)
    image_dir = args.output_dir / "images"
    seed_evidence = {"seeded_records": 0, "seeded_bytes": 0}
    if bool(config["reuse_verified_pilot"]):
        pilot_rows = [json.loads(line) for line in args.pilot_manifest.read_text(encoding="utf-8").splitlines() if line]
        seed_evidence = seed_verified_pilot(prepared, pilot_rows, image_dir)
    pixel_rows, near_pairs = download_pilot(
        prepared,
        image_dir=image_dir,
        maximum_total_bytes=int(config["maximum_total_bytes"]),
        timeout_seconds=int(config["timeout_seconds"]),
        max_retries=int(config["max_retries"]),
        request_interval_seconds=float(config["request_interval_seconds"]),
    )
    merged = merge_pixel_evidence(grouped, pixel_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    manifest = b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        for row in merged
    )
    manifest_path.write_bytes(manifest)
    payload_counts = Counter(row["sha256"] for row in pixel_rows)
    exact_duplicates = sum(count - 1 for count in payload_counts.values() if count > 1)
    total_bytes = sum(int(row["bytes"]) for row in pixel_rows)
    gates = config["gates"]
    checks = {
        "all_records_decode": len(pixel_rows) == int(config["expected_records"]),
        "exact_duplicate_payloads": exact_duplicates == int(gates["exact_duplicate_payloads"]),
        "near_duplicate_pairs": len(near_pairs)
        <= int(gates["maximum_near_duplicate_pairs_dhash_le_4"]),
        "maximum_total_bytes": total_bytes <= int(gates["maximum_total_bytes"]),
    }
    passed = all(checks.values())
    software_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": sha256(args.config),
        "grouping_decision_sha256": sha256(grouping_decision),
        "grouped_manifest_sha256": sha256(args.grouped_manifest),
        "phase_c_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "seed_evidence": seed_evidence,
        "records": len(pixel_rows),
        "known_creator_records": sum(bool(row["learning_eligible"]) for row in merged),
        "unknown_creator_stress_records": sum(not bool(row["learning_eligible"]) for row in merged),
        "total_bytes": total_bytes,
        "format_counts": dict(sorted(Counter(row["format"] for row in pixel_rows).items())),
        "mode_counts": dict(sorted(Counter(row["mode"] for row in pixel_rows).items())),
        "icc_profile_counts": dict(sorted(Counter("present" if row["icc_profile_bytes"] else "absent" for row in pixel_rows).items())),
        "exact_duplicate_payloads": exact_duplicates,
        "near_duplicate_pairs_dhash_le_4": near_pairs,
        "checks": checks,
        "passed": passed,
        "decision": "phase_c_pixels_complete" if passed else "phase_c_integrity_failed",
        "training_executed": False,
        "next_gate": config["next_gate"],
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    report_path = args.output_dir / "report.json"
    report_path.write_bytes(encoded)
    print(json.dumps({
        "decision": report["decision"],
        "records": report["records"],
        "total_bytes": total_bytes,
        "seeded_records": seed_evidence["seeded_records"],
        "exact_duplicates": exact_duplicates,
        "near_pairs": len(near_pairs),
        "manifest_sha256": report["phase_c_manifest_sha256"],
        "report_sha256": hashlib.sha256(encoded).hexdigest(),
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
