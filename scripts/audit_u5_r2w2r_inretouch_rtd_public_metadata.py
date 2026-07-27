#!/usr/bin/env python
"""Audit a retained public RTD repository tree without gated pixel access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.inretouch_rtd_source import (  # noqa: E402
    audit_inretouch_public_topology,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-api", type=Path, required=True)
    parser.add_argument("--local-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repository = json.loads(
        args.repository_api.read_text(encoding="utf-8")
    )
    with args.local_manifest.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "source_name" not in rows[0]:
        raise SystemExit("local manifest has no source_name rows")

    result = audit_inretouch_public_topology(
        repository, [row["source_name"] for row in rows]
    )
    report = {
        "schema_version": 1,
        "node": "ULT > U5 > U5.R2 > U5.R2W2R",
        "repository_api_sha256": _sha256(args.repository_api),
        "local_manifest_sha256": _sha256(args.local_manifest),
        "audit": result,
        "claim_ceiling": (
            "public repository-path topology and exact local FiveK identity "
            "overlap only; no gated pixel, operator, film or stock claim"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
