#!/usr/bin/env python3
"""Run U6.P8BO1 Android virtual-device runtime conformance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.native_standard_android_runtime import evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p8bo1_android_native_standard_runtime_v1.json",
    )
    parser.add_argument("--ndk", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--avd-home", type=Path, required=True)
    parser.add_argument("--port", type=int, default=5592)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        root=ROOT,
        contract_path=args.contract.resolve(),
        ndk=args.ndk.resolve(),
        sdk=args.sdk.resolve(),
        avd_home=args.avd_home.resolve(),
        output_dir=args.output_dir.resolve(),
        port=args.port,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes(
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
