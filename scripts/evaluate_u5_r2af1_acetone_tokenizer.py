#!/usr/bin/env python
"""Evaluate two exact U5.R2AF1 external-tokenizer manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.acetone_lut_tokenizer import evaluate_manifests, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--first-manifest", type=Path, required=True)
    parser.add_argument("--second-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = evaluate_manifests(
        load_manifest(args.first_manifest),
        load_manifest(args.second_manifest),
        config,
        root=Path.cwd(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], sort_keys=True))
    print(report["decision"])


if __name__ == "__main__":
    main()
