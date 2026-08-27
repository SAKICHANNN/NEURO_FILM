"""Run the frozen U4.3C three-stock face severe review materializer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.three_stock_face_severe_review import run_review


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u4_3c_three_stock_face_severe_review_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run_review(config, ROOT, args.output_dir, reverse=args.reverse)
    print(
        json.dumps(
            {"scientific_identity": report["scientific_identity"]}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
