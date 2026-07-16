"""Run the frozen SF1.0B connected-source identifiability audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.connected_stock_identifiability import load_connected_rows, run_connected_audit  # noqa: E402
from src.real_film.yfcc_stock_source import atomic_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "real_film_connected_stock_identifiability_v1.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run_connected_audit(load_connected_rows(ROOT, config), config)
    digest = atomic_json(ROOT / config["report_output"], report)
    print(json.dumps({
        "rows": report["rows"],
        "all_stock_edges_passed": report["all_stock_edges_passed"],
        "report_sha256": digest,
        "edges": {key: value["stock_signal_gate_passed"] for key, value in report["contrasts"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
