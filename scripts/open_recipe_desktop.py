#!/usr/bin/env python3
"""Open the private integrated three-look desktop workflow."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.recipe_desktop_workflow import IntegratedRecipeDesktopSession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a one-shot local browser for verified three-look recipe export."
    )
    parser.add_argument("history_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "configs/render_profiles/safe_rich_v1.json",
    )
    parser.add_argument("--tile-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with IntegratedRecipeDesktopSession(
        args.history_root,
        args.output_root,
        profile_path=args.profile,
        root=ROOT,
        tile_size=args.tile_size,
    ) as session:
        if os.name != "nt":
            raise RuntimeError("desktop browser launch currently requires Windows")
        os.startfile(session.url)  # type: ignore[attr-defined]
        result = session.wait()
    if result is None:
        return 1
    print(f"Exported {result.style}: {result.output_path}")
    print(f"SHA-256 {result.output_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
