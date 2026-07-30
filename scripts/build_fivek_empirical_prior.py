#!/usr/bin/env python3
"""Audit and freeze the research-only FiveK neutral-photo moment prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_match.research.fivek_prior_build import (  # noqa: E402
    build_fivek_empirical_prior_payload,
)
from src.inference import atomic_write_json  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root of the external neuro_film checkout containing freeze_v1.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload = build_fivek_empirical_prior_payload(args.dataset_root)
    artifact_sha256 = atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "prior_id": payload["prior_id"],
                "artifact_sha256": artifact_sha256,
                "source_image_count": payload["statistics"][
                    "source_image_count"
                ],
                "source_pixel_count": payload["statistics"][
                    "source_pixel_count"
                ],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
