"""Build and gate conservative creator/LOC-sequence groups for FSA/OWI."""

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

from src.real_film.fsa_owi_grouping import (  # noqa: E402
    build_sequence_guard_groups,
    evaluate_group_gate,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "real_film_fsa_owi_grouping.json"
    )
    parser.add_argument(
        "--canonical-manifest",
        type=Path,
        default=ROOT / "outputs" / "real_film" / "fsa_owi_v1" / "metadata" / "canonical_manifest.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "real_film" / "fsa_owi_v1" / "grouping",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if sha256(args.canonical_manifest) != config["canonical_manifest_sha256"]:
        raise ValueError("canonical manifest hash mismatch")
    pilot_decision = ROOT / "configs" / "real_film_fsa_owi_pilot_decision.json"
    if sha256(pilot_decision) != config["pilot_decision_sha256"]:
        raise ValueError("pilot decision hash mismatch")
    records = [json.loads(line) for line in args.canonical_manifest.read_text(encoding="utf-8").splitlines() if line]
    rows = build_sequence_guard_groups(
        records,
        maximum_adjacent_gap=int(config["group_method"]["maximum_adjacent_numeric_gap"]),
    )
    gate = evaluate_group_gate(rows, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    manifest = b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        for row in rows
    )
    manifest_path.write_bytes(manifest)
    software_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": sha256(args.config),
        "canonical_manifest_sha256": sha256(args.canonical_manifest),
        "pilot_decision_sha256": sha256(pilot_decision),
        "grouped_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "gate": gate,
        "group_interpretation": config["group_method"]["interpretation"],
        "evaluation_contract": config["evaluation_contract"],
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    report_path = args.output_dir / "report.json"
    report_path.write_bytes(encoded)
    print(json.dumps({
        "decision": gate["decision"],
        "records": gate["records"],
        "known_creator_records": gate["known_creator_records"],
        "sequence_groups": gate["sequence_group_count"],
        "report_sha256": hashlib.sha256(encoded).hexdigest(),
        "manifest_sha256": report["grouped_manifest_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
