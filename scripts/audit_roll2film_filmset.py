"""Freeze local FilmSet integrity, pair-blind pools, and evaluator lockboxes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roll2film.manifests import (  # noqa: E402
    FilmSetEvidenceConfig,
    FilmSetManifestError,
    build_filmset_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "data" / "raw" / "filmset" / "FilmSet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "roll2film" / "filmset_evidence",
    )
    return parser.parse_args()


def _commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> int:
    args = parse_args()
    try:
        result = build_filmset_evidence(
            FilmSetEvidenceConfig(
                root=args.root,
                output_dir=args.output_dir,
                software_commit=_commit(),
            ),
            progress=lambda message: print(message, flush=True),
        )
    except FilmSetManifestError as exc:
        print(json.dumps({"decision": "fail", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "decision": result.report["decision"],
                "report": str((args.output_dir / "report.json").resolve()),
                "pool_identity_counts": result.report["split_contract"]["pool_identity_counts"],
                "tree_sha256": result.report["archive_observed"]["tree_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
