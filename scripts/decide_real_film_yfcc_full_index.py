"""Adjudicate two byte-identical SF1.1 full-index metadata audits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.yfcc_full_index import decide_repeated_audits  # noqa: E402
from src.real_film.yfcc_stock_source import atomic_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "real_film_yfcc_full_index_v1.json")
    parser.add_argument("--report-a", type=Path, default=None)
    parser.add_argument("--report-b", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report_a = args.report_a or (ROOT / config["scan"]["report_a"])
    report_b = args.report_b or (ROOT / config["scan"]["report_b"])
    decision = decide_repeated_audits(report_a.read_bytes(), report_b.read_bytes(), config)
    digest = atomic_json(ROOT / config["scan"]["decision"], decision)
    print(json.dumps({"decision": decision["decision"], "passing_gate_ids": decision["passing_gate_ids"], "decision_sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
