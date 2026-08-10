"""Run the frozen U6.P6AM fixed-parameter scanner OECF diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.scanner_log_oecf_profile import canonical_json
from src.eval.scanner_step_wedge_transfer import evaluate, load_contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p6am_barnard_step_wedge_transfer_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(load_contract(args.config, ROOT), ROOT)
    payload = canonical_json(report) + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(hashlib.sha256(payload).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
