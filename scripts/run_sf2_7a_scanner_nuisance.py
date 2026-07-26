#!/usr/bin/env python
"""Run the frozen SF2.7A scanner/software nuisance quantification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.scanner_nuisance import build_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/sf2_7a_scanner_nuisance_quantification_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/sf2_7a_scanner_nuisance_quantification_v1/report.json",
    )
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(ROOT, config)
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "all_alignment_gates_passed": report[
                    "all_alignment_gates_passed"
                ],
                "aggregate": report["aggregate"],
                "interpretation": report["interpretation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
