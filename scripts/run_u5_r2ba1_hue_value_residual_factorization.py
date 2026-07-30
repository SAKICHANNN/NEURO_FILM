from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.b0_hue_value_residual_factorization import (
    render_and_evaluate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/u5_r2ba1_hue_value_residual_factorization_v1.json",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    report = render_and_evaluate(
        root=ROOT,
        config=config,
        output_dir=ROOT / args.output_dir,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")


if __name__ == "__main__":
    main()
