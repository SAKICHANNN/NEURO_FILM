#!/usr/bin/env python3
"""Freeze metadata or acquire the exact U6.P4Z B&W grain source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.bw_uniform_grain_source import (  # noqa: E402
    acquire_files,
    fetch_metadata_snapshot,
)
from src.eval.real_uniform_grain_source import write_atomic_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p4z_bw_uniform_grain_source_v1.json",
    )
    parser.add_argument(
        "--mode",
        choices=("metadata", "acquire"),
        default="metadata",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.mode == "metadata":
        payload = fetch_metadata_snapshot(config)
        destination = ROOT / str(config["outputs"]["metadata_snapshot"])
    else:
        payload = acquire_files(ROOT, config)
        destination = ROOT / str(config["outputs"]["acquisition_manifest"])
    digest = write_atomic_json(destination, payload)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "path": str(destination),
                "rows": len(payload["rows"]),
                "sha256": digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
