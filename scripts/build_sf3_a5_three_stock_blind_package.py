"""Build the severe-gated, hash-bound SF3.A5 blind package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.three_stock_blind_package import build_package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/sf3_a5_three_stock_blind_distinguishability_v1.json",
    )
    parser.add_argument("--render-report", type=Path, action="append", required=True)
    parser.add_argument("--render-root", type=Path, action="append", required=True)
    parser.add_argument("--severe-review", type=Path, action="append", required=True)
    parser.add_argument("--secret-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    secret = args.secret_file.read_text(encoding="utf-8").strip()
    report = build_package(
        args.contract,
        root=ROOT,
        render_report_path=(
            args.render_report[0] if len(args.render_report) == 1 else args.render_report
        ),
        render_root=args.render_root[0] if len(args.render_root) == 1 else args.render_root,
        severe_review_path=(
            args.severe_review[0]
            if len(args.severe_review) == 1
            else args.severe_review
        ),
        secret=secret,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("automatic_pass", "decision", "stable_evidence_id")
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
