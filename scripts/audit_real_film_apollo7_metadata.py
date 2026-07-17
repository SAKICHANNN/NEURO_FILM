"""Run the frozen SF2.0A Apollo 7 bounded HTML metadata feasibility audit."""

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

from src.real_film.apollo_metadata_feasibility import run_metadata_feasibility_audit  # noqa: E402
from src.real_film.yfcc_stock_source import atomic_json  # noqa: E402


def _software_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "real_film_apollo7_metadata_feasibility_v1.json",
    )
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    report, decision = run_metadata_feasibility_audit(config)
    provenance = {
        "config_path": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "software_commit": _software_commit(),
    }
    report["provenance"] = provenance
    decision["provenance"] = provenance
    report_sha = atomic_json(ROOT / config["report"], report)
    decision["report_sha256"] = report_sha
    decision_sha = atomic_json(ROOT / config["decision"], decision)
    print(
        json.dumps(
            {
                "page_requests": report["page_requests"],
                "valid_pages": decision["valid_pages"],
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
