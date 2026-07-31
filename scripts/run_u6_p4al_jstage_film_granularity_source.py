#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.jstage_film_granularity_source import acquire_and_evaluate, load_contract  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/u6_p4al_jstage_film_granularity_source_v1.json")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = acquire_and_evaluate(load_contract(args.config), ROOT)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(args.report.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    with temporary.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, args.report)
    print(json.dumps({"branch": report["branch"], "source_pass": report["source_pass"], "stable_evidence_id": report["stable_evidence_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
