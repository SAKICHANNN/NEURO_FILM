#!/usr/bin/env python3
"""Compile and replay the frozen U6.P8A fixed-reference profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.film_physics.profile_compiler import (  # noqa: E402
    evaluate_compiled_profile,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs"
        / "u6_p8a_fixed_reference_profile_compiler_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = evaluate_compiled_profile(root=ROOT, config=config)
    digest = write_report(report, args.output)
    print(f"decision={report['decision']}")
    print(f"bundle_sha256={report['artifact']['bundle_sha256']}")
    print(f"artifact_sha256={report['artifact_canonical_sha256']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
