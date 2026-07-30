"""Build or adjudicate the AX17 fixed-baseline blind comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.filmmatch_baseline_blind import (  # noqa: E402
    adjudicate_filmmatch_blind,
    build_filmmatch_blind_sheets,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2ax17_filmmatch_fixed_baseline_blind_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "outputs/external_controls/filmmatch_ektachrome_v1/"
        "fixed_baseline_blind_ax17",
    )
    parser.add_argument("--observations", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.observations is None:
        receipt = build_filmmatch_blind_sheets(
            root=ROOT, config=config, output_dir=args.output_dir
        )
        print(
            json.dumps(
                {
                    "rounds": len(receipt["artifacts"]),
                    "stable_evidence_id": receipt["stable_evidence_id"],
                },
                sort_keys=True,
            )
        )
        return 0
    report = adjudicate_filmmatch_blind(
        root=ROOT,
        config=config,
        build_dir=args.output_dir,
        observations_path=args.observations,
    )
    report_path = args.output_dir / "adjudication.json"
    temporary = report_path.with_name("adjudication.tmp.json")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
    print(
        json.dumps(
            {
                "blind_gate_passed": report["blind_gate_passed"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
