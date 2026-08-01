#!/usr/bin/env python3
"""Run the frozen U6.P5N positive residual compiler."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_positive_mtf_residual import compile_and_evaluate_residual


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p5n_positive_residual_compiler_v1.json",
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    bundle, report = compile_and_evaluate_residual(
        root=ROOT, config=config, diagnostic_path=args.diagnostic.resolve()
    )
    _write(args.bundle, bundle)
    _write(args.report, report)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "residual_bundle_id": report["residual_bundle_id"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
