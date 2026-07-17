"""Audit a frozen SF2.1A Openverse snapshot without network or pixels."""

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

from src.real_film.openverse_stock_source import audit_snapshot  # noqa: E402
from src.real_film.yfcc_stock_source import atomic_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "real_film_openverse_shared_creator_v1.json"
    )
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    snapshot_path = args.snapshot or ROOT / config["snapshot"]
    snapshot_bytes = snapshot_path.read_bytes()
    snapshot = json.loads(snapshot_bytes)
    report, decision = audit_snapshot(snapshot, config)
    repeated_report, repeated_decision = audit_snapshot(snapshot, config)
    if report != repeated_report or decision != repeated_decision:
        raise RuntimeError("offline audit is not repeat-identical")
    provenance = {
        "config_path": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "snapshot_path": str(snapshot_path.resolve()),
        "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
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
                "selected_component": decision["selected_component"],
                "report_sha256": report_sha,
                "decision_sha256": decision_sha,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
