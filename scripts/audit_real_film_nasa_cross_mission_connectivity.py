"""Run frozen SF2.0C0 NASA/JSC cross-mission stock connectivity census."""

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

from src.real_film.nasa_sts_stock_snapshot import run_cross_mission_census  # noqa: E402
from src.real_film.yfcc_stock_source import atomic_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "real_film_nasa_cross_mission_connectivity_v1.json",
    )
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    report, decision = run_cross_mission_census(config)
    provenance = {
        "config_path": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
        ).strip(),
    }
    report["provenance"] = provenance
    decision["provenance"] = provenance
    report_sha = atomic_json(ROOT / config["report"], report)
    decision["report_sha256"] = report_sha
    decision_sha = atomic_json(ROOT / config["decision"], decision)
    print(
        json.dumps(
            {
                "requests_made": report["requests_made"],
                "decision": decision["decision"],
                "selected_mission": decision["selected_mission"],
                "candidate_missions": decision["candidate_missions"],
                "report_sha256": report_sha,
                "decision_sha256": decision_sha,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
