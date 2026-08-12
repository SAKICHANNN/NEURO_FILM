#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.portable_native_spatial_density_runtime import evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--host-clang", type=Path, required=True)
    parser.add_argument("--ndk", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--avd-home", type=Path, required=True)
    parser.add_argument("--avd-name", required=True)
    parser.add_argument("--port", type=int, default=5582)
    args = parser.parse_args()
    result = evaluate(
        root=ROOT,
        contract=json.loads(args.contract.read_text("utf-8")),
        output_dir=args.output_dir,
        host_clang=args.host_clang,
        ndk=args.ndk,
        sdk=args.sdk,
        avd_home=args.avd_home,
        avd_name=args.avd_name,
        port=args.port,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", "utf-8")
    print(json.dumps(result["stable"], sort_keys=True))
    return 0 if result["stable"]["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
