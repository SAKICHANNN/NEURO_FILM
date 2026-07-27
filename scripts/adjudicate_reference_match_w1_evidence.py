#!/usr/bin/env python3
"""Adjudicate stable external W1 reports without importing external code."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_match.research.w1_evidence import (  # noqa: E402
    inspect_w1_development_evidence,
    load_w1_intake_contract,
    w1_evidence_decision_to_json,
)
from src.inference import atomic_write_json  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT
        / "configs"
        / "reference_match_w1_development_intake_v1.json",
    )
    parser.add_argument("--external-repo", type=Path, required=True)
    parser.add_argument("--report-a", type=Path, required=True)
    parser.add_argument("--report-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    contract = load_w1_intake_contract(args.contract)
    decision = inspect_w1_development_evidence(
        contract,
        external_repo=args.external_repo,
        report_a=args.report_a,
        report_b=args.report_b,
    )
    atomic_write_json(
        args.output,
        w1_evidence_decision_to_json(decision),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
