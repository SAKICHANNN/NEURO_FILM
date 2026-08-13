"""Run one fresh process of the frozen P4HZ evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.opponent_diffusion_photographic_confirmation import (
    evaluate,
    load_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs"
        / "u6_p4hz_opponent_diffusion_photographic_confirmation_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    contract = load_contract(arguments.config)
    result = evaluate(contract, root=ROOT, output_dir=arguments.output_dir)
    report = arguments.output_dir / "report.json"
    report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(report),
                "automatic_pass": result["automatic_pass"],
                "decision": result["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
