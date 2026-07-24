#!/usr/bin/env python3
"""Render/evaluate the frozen U5.R2I1B neutral-gauged frontier."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_u5_r2i0_sensitometry_print_frontier import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(ROOT / "configs/u5_r2i1b_neutral_gauge_frontier_v1.json"))
