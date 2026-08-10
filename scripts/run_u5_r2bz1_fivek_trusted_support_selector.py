"""Run the frozen U5.R2BZ1 trusted-support selector control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.fivek_proxy_exploitation_control import canonical_bytes
from src.eval.fivek_trusted_support_selector import run_control


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2bz1_fivek_trusted_support_selector_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("U5.R2BZ1 report is create-only")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(run_control(ROOT, config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
