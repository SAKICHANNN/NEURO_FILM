#!/usr/bin/env python3
"""Run the frozen U6.P4CG Boolean/Gaussian LOD diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.boolean_gaussian_lod_boundary import (
    evaluate_boolean_gaussian_lod_boundary,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p4cg_boolean_gaussian_lod_boundary_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    report = evaluate_boolean_gaussian_lod_boundary(contract, ROOT)
    digest = write_report(report, args.output)
    print(json.dumps({"report_sha256": digest, "decision": report["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
