#!/usr/bin/env python3
"""Print a validated render profile's declared evidence without inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import (  # noqa: E402
    RenderContractError,
    load_render_profile,
    summarize_render_profile_evidence,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a K-MCFM render profile and print its declared evidence ledger."
    )
    parser.add_argument(
        "profile",
        nargs="?",
        type=Path,
        default=ROOT / "configs" / "render_profiles" / "safe_rich_v1.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        profile = load_render_profile(args.profile, root=ROOT)
        summary = summarize_render_profile_evidence(profile)
    except (OSError, json.JSONDecodeError, RenderContractError) as error:
        print(f"profile inspection failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
