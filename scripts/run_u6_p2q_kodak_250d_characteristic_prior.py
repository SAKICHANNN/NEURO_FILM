#!/usr/bin/env python
"""Compile and audit the frozen Kodak 250D characteristic prior."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_characteristic_prior import evaluate_prior, load_contract


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p2q_kodak_250d_characteristic_prior_v1.json",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    report, bundle = evaluate_prior(load_contract(args.config), ROOT)
    _write_json(args.report, report)
    _write_json(args.bundle, bundle)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "decision": report["decision"],
                "stable_evidence_id": report["stable_evidence_id"],
                "prior_identity": report["prior_identity"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
