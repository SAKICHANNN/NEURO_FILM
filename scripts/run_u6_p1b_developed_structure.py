#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_developed_structure import (  # noqa: E402
    evaluate_developed_structure,
    load_contract,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/u6_p1b_developed_structure_reference_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_developed_structure(load_contract(args.contract))
    digest = write_report(report, args.output)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")


if __name__ == "__main__":
    main()
