"""Match one uploaded reference across N source images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_match import (  # noqa: E402
    ReferenceMatchContractError,
    build_file_match_report,
    match_reference_files,
    save_file_match_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one deterministic reference-look recipe and apply it to N "
            "SDR source images."
        )
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        help="Source image; repeat once per input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        action="append",
        required=True,
        help="Output image; repeat in source order.",
    )
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--bit-depth",
        type=int,
        choices=(8, 16),
        default=16,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = match_reference_files(
            args.reference,
            args.source,
            args.output,
            recipe_path=args.recipe,
            output_bit_depth=args.bit_depth,
        )
        report_sha256 = save_file_match_report(result, args.report)
    except (OSError, ReferenceMatchContractError, ValueError) as exc:
        print(f"reference match failed: {exc}", file=sys.stderr)
        return 2
    report = build_file_match_report(result)
    summary = {
        "schema_id": report["schema_id"],
        "recipe_id": report["recipe_id"],
        "report_path": str(args.report.resolve()),
        "report_sha256": report_sha256,
        "output_count": len(report["outputs"]),
        "applied_count": sum(
            row["safety"]["accepted"] for row in report["outputs"]
        ),
        "identity_fallback_count": sum(
            not row["safety"]["accepted"] for row in report["outputs"]
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
