#!/usr/bin/env python3
"""Batch-evaluate the deterministic color pipeline with a named profile."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_color_baseline import main as audit_main  # noqa: E402


def main() -> int:
    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--output-dir",
                "outputs/eval/baseline_saferich",
                "--profile",
                "configs/color_rendering_profiles.yaml",
                "--profile-name",
                "safe_rich",
            ]
        )
    return audit_main()


if __name__ == "__main__":
    raise SystemExit(main())
