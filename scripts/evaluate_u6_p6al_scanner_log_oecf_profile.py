from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.scanner_log_oecf_profile import canonical_json, evaluate, load_contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/u6_p6al_scanner_log_oecf_profile_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_contract(root / args.config, root)
    report = evaluate(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
