from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.velvia_vision3_joint_source_signature import (
    audit_joint_source_signature,
    load_contract,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2bw1_velvia_vision3_joint_source_signature_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_joint_source_signature(load_contract(args.config), ROOT)
    write_report(report, args.output)
    return 0 if report["decision"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
