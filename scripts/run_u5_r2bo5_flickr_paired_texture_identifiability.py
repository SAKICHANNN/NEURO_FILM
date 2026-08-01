from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.flickr_paired_texture_identifiability import evaluate
from src.eval.flickr_single_author_pair_acquisition import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/u5_r2bo5_flickr_paired_texture_identifiability_v1.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    atomic_json(root / args.output, evaluate(root, config))


if __name__ == "__main__":
    main()
