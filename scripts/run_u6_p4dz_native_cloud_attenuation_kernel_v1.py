from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.native_cloud_attenuation_conformance import evaluate


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--build-dir", type=Path, required=True)
    p.add_argument(
        "--clang",
        type=Path,
        default=ROOT
        / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe",
    )
    a = p.parse_args()
    r = evaluate(
        ROOT,
        ROOT / "configs/u6_p4dz_native_cloud_attenuation_kernel_v1.json",
        a.build_dir,
        a.clang,
    )
    a.output.write_text(json.dumps(r, indent=2, sort_keys=True) + "\n")
    return 0 if r["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
