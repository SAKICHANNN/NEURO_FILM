"""CLI runner for U5.R2CB3."""

from __future__ import annotations

import argparse
from pathlib import Path

from .fujifilm_e6_dye_operator_photographic import (
    evaluate_photographic,
    load_contract,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = evaluate_photographic(
        load_contract(args.config), args.root.resolve(), args.output_dir
    )
    report_sha256 = write_report(report, args.output)
    print(f"decision={report['decision']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={report_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
