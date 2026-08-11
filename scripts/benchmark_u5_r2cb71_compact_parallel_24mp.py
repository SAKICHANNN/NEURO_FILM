#!/usr/bin/env python3
"""Run the CB70 resource harness with the compact exact CB71 renderer."""

from __future__ import annotations

from pathlib import Path

import scripts.benchmark_u5_r2cb70_parallel_target_24mp as harness
from src.eval.analytic_y_chromaticity_compact_parallel_candidate import (
    render_analytic_y_chromaticity_compact_parallel_candidate,
)

# The reused parent launches its own module path. Point that path at this wrapper so
# fresh child processes retain the CB71 renderer substitution.
harness.__file__ = str(Path(__file__).resolve())
harness.render_analytic_y_chromaticity_parallel_target_candidate = (
    render_analytic_y_chromaticity_compact_parallel_candidate
)


if __name__ == "__main__":
    raise SystemExit(harness.main())
