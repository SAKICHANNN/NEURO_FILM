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
    ReferenceRenderGuardPolicy,
    build_file_match_report,
    build_file_replay_report,
    match_reference_files,
    replay_reference_files,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one deterministic reference-look recipe or replay one "
            "verified recipe across N SDR source images."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--reference",
        type=Path,
        help="Reference image used to fit a new recipe.",
    )
    mode.add_argument(
        "--recipe-input",
        type=Path,
        help="Existing verified recipe replayed without the reference file.",
    )
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
        help=(
            "Output image; repeat in source order. Linear Rec.2020 SDR "
            "sources require 16-bit PNG and retain BT.2020 CICP."
        ),
    )
    parser.add_argument(
        "--recipe",
        type=Path,
        help="Required output recipe path in --reference mode.",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--bit-depth",
        type=int,
        choices=(8, 16),
        default=16,
        help=(
            "Output precision. Rec.2020 SDR supports 16-bit PNG only; "
            "sRGB also supports 8-bit PNG/JPEG/TIFF and 16-bit PNG/TIFF."
        ),
    )
    parser.add_argument(
        "--allow-research-baseline",
        action="store_true",
        help=(
            "Apply the rejected safe-Lab research baseline when pixel guards "
            "pass. Without this explicit override, delivery is identity."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.reference is not None and args.recipe is None:
        print(
            "reference match failed: --reference mode requires --recipe",
            file=sys.stderr,
        )
        return 2
    if args.recipe_input is not None and args.recipe is not None:
        print(
            "reference match failed: --recipe is output-only and cannot be "
            "used with --recipe-input",
            file=sys.stderr,
        )
        return 2
    guard_policy = ReferenceRenderGuardPolicy(
        allow_research_baseline=args.allow_research_baseline,
    )
    try:
        if args.reference is not None:
            result = match_reference_files(
                args.reference,
                args.source,
                args.output,
                recipe_path=args.recipe,
                report_path=args.report,
                output_bit_depth=args.bit_depth,
                guard_policy=guard_policy,
            )
            report_sha256 = result.report_file_sha256
            report = build_file_match_report(result)
            operation = "fit-and-render"
        else:
            replay_result = replay_reference_files(
                args.recipe_input,
                args.source,
                args.output,
                report_path=args.report,
                output_bit_depth=args.bit_depth,
                guard_policy=guard_policy,
            )
            report_sha256 = replay_result.report_file_sha256
            report = build_file_replay_report(replay_result)
            operation = "recipe-replay"
    except (OSError, ReferenceMatchContractError, ValueError) as exc:
        print(f"reference match failed: {exc}", file=sys.stderr)
        return 2
    if report_sha256 is None:  # pragma: no cover - CLI invariant.
        print(
            "reference match failed: transactional report was not committed",
            file=sys.stderr,
        )
        return 2
    summary = {
        "schema_id": report["schema_id"],
        "operation": operation,
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
