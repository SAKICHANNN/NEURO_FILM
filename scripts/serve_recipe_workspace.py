#!/usr/bin/env python3
"""Serve one local browser-driven strict recipe export session."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.recipe_browser_export import RecipeBrowserExportSession


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "configs/render_profiles/safe_rich_v1.json",
    )
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    with RecipeBrowserExportSession(
        args.history_root,
        args.output_root,
        profile_path=args.profile,
        root=ROOT,
        tile_size=args.tile_size,
    ) as session:
        print(json.dumps({"status": "READY", "url": session.url}), flush=True)
        if not args.no_open:
            webbrowser.open(session.url)
        result = session.wait(args.timeout)
        if result is None:
            print(json.dumps({"status": "TIMEOUT"}, sort_keys=True))
            return 1
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "style": result.style,
                    "output_path": str(result.output_path),
                    "output_sha256": result.output_sha256,
                    "receipt": result.receipt,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
