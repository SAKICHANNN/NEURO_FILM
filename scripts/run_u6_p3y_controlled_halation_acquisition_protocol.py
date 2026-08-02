"""Validate the compiled U6.P3Y controlled halation acquisition protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.controlled_halation_acquisition_protocol import (
    validate_protocol,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT
        / "configs/u6_p3y_controlled_halation_acquisition_protocol_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_protocol(ROOT, args.protocol)
    digest = write_report(report, args.output)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "protocol_stable_id": report["protocol_stable_id"],
                "report_sha256": digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
