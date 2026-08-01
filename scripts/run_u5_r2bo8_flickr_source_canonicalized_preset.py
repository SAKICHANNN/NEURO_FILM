from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.flickr_source_canonicalized_preset import evaluate  # noqa: E402
from src.eval.flickr_single_author_pair_acquisition import atomic_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/u5_r2bo8_flickr_source_canonicalized_preset_v1.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    atomic_json(ROOT / args.output, evaluate(ROOT, config))


if __name__ == "__main__":
    main()
