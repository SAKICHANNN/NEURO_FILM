"""Audit a bounded current Openverse snapshot after excluding prior identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.openverse_three_stock_refresh import (
    audit_fresh_snapshot,
)
from src.real_film.yfcc_stock_source import atomic_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sf3_a2_openverse_three_stock_refresh_v1.json",
    )
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    snapshot_path = args.snapshot or ROOT / config["snapshot"]
    snapshot_bytes = snapshot_path.read_bytes()
    snapshot = json.loads(snapshot_bytes)
    report, decision = audit_fresh_snapshot(snapshot, config, root=ROOT)
    repeated = audit_fresh_snapshot(snapshot, config, root=ROOT)
    if repeated != (report, decision):
        raise RuntimeError("offline refresh audit is not repeat-identical")
    provenance = {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
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
                "report_sha256": report_sha,
                "decision_sha256": decision_sha,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
