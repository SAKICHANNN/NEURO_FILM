#!/usr/bin/env python
"""Run the frozen U6.P3J streamed FFT backing-return evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_streamed_fft_backing_return import (  # noqa: E402
    P3D_SCHEMA,
    P3J_SCHEMA,
    evaluate_streamed_fft_backing_return,
    load_json,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent",
        type=Path,
        default=ROOT / "configs" / "u6_p3d_backing_return_reference_v1.json",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "u6_p3j_streamed_fft_backing_return_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_streamed_fft_backing_return(
        load_json(args.parent, P3D_SCHEMA),
        load_json(args.contract, P3J_SCHEMA),
    )
    digest = write_report(report, args.output)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")


if __name__ == "__main__":
    main()
