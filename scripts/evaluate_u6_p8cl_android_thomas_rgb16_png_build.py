#!/usr/bin/env python3
"""Run U6.P8CL Android dual-ABI build conformance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.native_thomas_rgb16_png_android_conformance import evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p8cl_android_thomas_rgb16_png_build_v1.json",
    )
    parser.add_argument("--ndk", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/experiments/u6_p8cl_android_thomas_rgb16_png_build_v1",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = evaluate(args.contract.resolve(), args.ndk.resolve(), args.output_dir.resolve())
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if args.report is None:
        sys.stdout.buffer.write(encoded)
    else:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(encoded)
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
