from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.native_cloud_attenuation_android_runtime import evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        ROOT,
        ROOT / "configs/u6_p4ec_native_cloud_attenuation_android_tolerance_v2.json",
        Path(r"D:\nf-019f4b76-android\android-ndk-r27d"),
        ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe",
        Path(r"D:\nf-019f4b76-android\runtime-sdk"),
        Path(r"D:\nf-019f4b76-android\avd"),
        args.work,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
