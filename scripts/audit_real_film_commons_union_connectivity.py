"""Run the frozen SF2.3 Commons-union metadata connectivity audit."""

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

from src.real_film.commons_union_connectivity import audit_union  # noqa: E402
from src.real_film.yfcc_stock_source import atomic_json  # noqa: E402


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "real_film_commons_union_connectivity_v1.json",
    )
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    inputs = []
    provenance_inputs = []
    hash_errors = []
    for index, item in enumerate(config["inputs"]):
        snapshot_path = ROOT / item["snapshot"]
        source_config_path = ROOT / item["source_config"]
        snapshot_bytes = snapshot_path.read_bytes()
        source_config_bytes = source_config_path.read_bytes()
        snapshot_sha = _sha256(snapshot_bytes)
        source_config_sha = _sha256(source_config_bytes)
        if snapshot_sha != item["snapshot_sha256"]:
            hash_errors.append(f"input-{index}:snapshot-sha256-mismatch")
        if source_config_sha != item["source_config_sha256"]:
            hash_errors.append(f"input-{index}:source-config-sha256-mismatch")
        inputs.append((json.loads(snapshot_bytes), json.loads(source_config_bytes)))
        provenance_inputs.append(
            {
                "snapshot": item["snapshot"],
                "snapshot_sha256": snapshot_sha,
                "source_config": item["source_config"],
                "source_config_sha256": source_config_sha,
            }
        )
    report, decision = audit_union(inputs, config, input_contract_errors=hash_errors)
    repeated_report, repeated_decision = audit_union(inputs, config, input_contract_errors=hash_errors)
    if (report, decision) != (repeated_report, repeated_decision):
        raise RuntimeError("offline audit is not repeat-identical")
    provenance = {
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha256(config_bytes),
        "inputs": provenance_inputs,
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
        ).strip(),
        "offline_repeat_identical": True,
    }
    report["provenance"] = provenance
    decision["provenance"] = provenance
    report_sha = atomic_json(ROOT / config["report"], report)
    decision["report_sha256"] = report_sha
    decision_sha = atomic_json(ROOT / config["decision"], decision)
    print(
        json.dumps(
            {
                "decision": decision["decision"],
                "eligible_stocks": report["eligible_stocks"],
                "retained_edges": len(report["retained_edges"]),
                "selected_component": decision["selected_component"],
                "report_sha256": report_sha,
                "decision_sha256": decision_sha,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
