#!/usr/bin/env python3
"""Audit a legacy image manifest for FilmCase lineage and split eligibility."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Support direct invocation from the repository root without requiring callers
# to set PYTHONPATH.  The FilmCase module itself remains a normal ``src``
# package and does not modify global interpreter state.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.filmcase.lineage import ManifestAuditError, audit_manifest, write_audit_outputs


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "manifest.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "filmcase" / "u03_lineage_audit"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an isolated FilmCase manifest-v2 audit without changing the legacy manifest."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Legacy JSONL manifest to inspect.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Ignored output directory for report.json and manifest_v2.jsonl.",
    )
    parser.add_argument("--validation-percent", type=int, default=10)
    parser.add_argument("--perceptual-threshold", type=int, default=4)
    parser.add_argument("--skip-perceptual", action="store_true", help="Skip local dHash computation.")
    parser.add_argument("--write", action="store_true", help="Write audit outputs; otherwise print report only.")
    parser.add_argument(
        "--fail-on-unresolved",
        action="store_true",
        help="Return nonzero unless at least one source-group-resolved reference row is eligible.",
    )
    args = parser.parse_args()

    try:
        result = audit_manifest(
            args.input,
            root=ROOT,
            validation_percent=args.validation_percent,
            perceptual_threshold=args.perceptual_threshold,
            compute_perceptual_hashes=not args.skip_perceptual,
        )
    except ManifestAuditError as exc:
        parser.error(str(exc))

    if args.write:
        write_audit_outputs(
            result,
            report_path=args.output_dir / "report.json",
            manifest_v2_path=args.output_dir / "manifest_v2.jsonl",
        )
    print(json.dumps(result.report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.fail_on_unresolved and result.report["gate_status"] != "ready_for_group_split":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
