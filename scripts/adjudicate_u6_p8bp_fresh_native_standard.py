#!/usr/bin/env python3
"""Adjudicate the frozen U6.P8BP fixed-arm blind comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.fresh_native_standard_adjudication import (  # noqa: E402
    adjudicate_fresh_native_standard,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs"
        / "u6_p8bp_fresh_native_standard_adjudication_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs"
        / "u6_p8bp_fresh_native_standard_confirmation_v1"
        / "adjudication.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = adjudicate_fresh_native_standard(
        root=ROOT,
        config=config,
    )
    digest = write_report(report, args.output)
    print(f"decision={report['decision']}")
    print(
        "native_round_wins="
        f"{report['native_vs_ao6_round_wins']['fixed_native_standard_full_strength_1_0']}"
    )
    print(
        "ao6_round_wins="
        f"{report['native_vs_ao6_round_wins']['fixed_ao6_colour_only_t15_c35']}"
    )
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
