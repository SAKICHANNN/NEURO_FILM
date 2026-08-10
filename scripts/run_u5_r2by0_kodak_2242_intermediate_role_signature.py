"""Run the frozen U5.R2BY0 intermediate-material source audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.kodak_intermediate_role_signature import (
    audit_intermediate_role_signature,
    load_contract,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/u5_r2by0_kodak_2242_intermediate_role_signature_v1.json"),
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_contract(root / args.config)
    report = audit_intermediate_role_signature(
        config, root, overlay_path=root / args.overlay
    )
    write_report(report, root / args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
