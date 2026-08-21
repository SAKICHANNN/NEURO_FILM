#!/usr/bin/env python3
"""Run the frozen P96 DNG camera-to-PCS portable ABI audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.dng_camera_to_pcs_native_conformance import evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    args = parser.parse_args()
    if args.output_root.exists() or args.report.exists():
        raise RuntimeError("P96 formal output path already exists")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = evaluate(
        root=ROOT,
        config=config,
        output_dir=args.output_root,
        order=args.order,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(report["stable_identity_sha256"])
    return 0 if report["decision"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
