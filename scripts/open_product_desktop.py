#!/usr/bin/env python3
"""Open the native new-input Look Approximation desktop workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve_product_scratch_root(candidate: Path, *, project_root: Path = ROOT) -> Path:
    """Resolve one desktop scratch root within the repository storage binding."""

    canonical = (Path(project_root).resolve(strict=True) / "tmp").resolve(strict=True)
    resolved = Path(candidate).resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("desktop scratch root must be an existing directory")
    try:
        resolved.relative_to(canonical)
    except ValueError as exc:
        raise ValueError(
            "desktop scratch root must be the repository tmp root or its descendant"
        ) from exc
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open the private new-input K-MCFM Look Approximation desktop."
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=ROOT / "tmp",
        help="Existing repository-relative generated-artifact root.",
    )
    parser.add_argument(
        "--smoke-exit-ms",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke_exit_ms is not None and args.smoke_exit_ms < 1:
        raise ValueError("--smoke-exit-ms must be positive")
    try:
        scratch_root = resolve_product_scratch_root(args.scratch_root)
    except (OSError, ValueError) as exc:
        print(f"K-MCFM desktop rejected: {exc}", file=sys.stderr)
        return 2
    import tkinter as tk

    from src.inference.product_desktop import ProductDesktopWorkflow
    from src.inference.product_desktop_ui import build_product_desktop_app

    root = tk.Tk()
    workflow = ProductDesktopWorkflow(root=ROOT, scratch_root=scratch_root)
    app = build_product_desktop_app(root, workflow, initial_input=args.input)
    if args.smoke_exit_ms is not None:
        root.after(args.smoke_exit_ms, app.close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
