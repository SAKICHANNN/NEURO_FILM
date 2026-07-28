#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_compiled_scatter import P1_SCHEMA, P3_SCHEMA, load_json  # noqa: E402
from src.eval.physical_pyramid_scatter import (  # noqa: E402
    PYRAMID_SCHEMA,
    evaluate_pyramid_scatter,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("configs/u6_p1_reference_scatter_simulator_v1.json"),
    )
    parser.add_argument(
        "--direct",
        type=Path,
        default=Path("configs/u6_p3_compiled_scatter_challenger_v1.json"),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/u6_p3_pyramid_scatter_challenger_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_pyramid_scatter(
        load_json(args.parent, P1_SCHEMA),
        load_json(args.direct, P3_SCHEMA),
        load_json(args.contract, PYRAMID_SCHEMA),
    )
    digest = write_report(report, args.output)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")


if __name__ == "__main__":
    main()
